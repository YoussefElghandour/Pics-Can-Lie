"""
Wikipedia Fact-Checking Module
================================
Pipeline: Caption -> spaCy NER -> Wikipedia API -> DeBERTa NLI -> Entailment Score

factcheck_score = entailment - contradiction  (range: -1 to +1)
  +1  strong entailment  (caption well-supported by Wikipedia)
   0  neutral / no evidence
  -1  strong contradiction (caption conflicts with Wikipedia)
"""
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402


import os, sys, time, json, re, logging, subprocess
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
WIKI_API        = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS    = {"User-Agent": "PicsCanLie-ThesisBot/1.0"}
WIKI_TIMEOUT    = 8
WIKI_RETRIES    = 2
WIKI_MAX_CHARS  = 1500
ENTITY_DELAY    = 0.2          # seconds between Wikipedia entity queries
MAX_ENTITIES    = 4
NLI_MODEL_ID    = "cross-encoder/nli-deberta-v3-large"
NER_TYPES       = {"PERSON", "ORG", "GPE", "EVENT", "LOC", "FAC", "NORP"}
CHECKPOINT_EVERY = 100

PROJECT_ROOT = Path(str(_cfg.ROOT))
OUT_CSV      = PROJECT_ROOT / "mmfakebench_factcheck_scores.csv"
CKPT_CSV     = PROJECT_ROOT / "mmfakebench_factcheck_checkpoint.csv"

# Val JSON search paths (checked in order)
VAL_SEARCH_PATHS = [
    PROJECT_ROOT / "MMFakeBench" / "val.json",
    PROJECT_ROOT / "MMFakeBench" / "val.jsonl",
    PROJECT_ROOT / "MMFakeBench" / "data" / "val.json",
    PROJECT_ROOT / "dataset" / "MMFakeBench" / "MMFakeBench_val.json",
    PROJECT_ROOT / "dataset" / "MMFakeBench" / "val.json",
]
CAPTION_COLS = ["caption", "text", "claim", "sentence", "title"]
LABEL_COLS   = ["label", "falsified", "fake", "is_fake", "type", "gt_answers"]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  spaCy — auto-install if missing
# ══════════════════════════════════════════════════════════════════════════════
def load_spacy():
    try:
        import spacy
    except ImportError:
        log.info("spaCy not found — installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy", "-q"])
        import spacy

    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        log.info("en_core_web_lg not found — downloading...")
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_lg"]
        )
        import spacy
        nlp = spacy.load("en_core_web_lg")

    log.info("spaCy en_core_web_lg loaded.")
    return nlp


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Named Entity Extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_entities(caption: str, nlp) -> list[str]:
    """Return deduplicated named entities (capped at MAX_ENTITIES)."""
    if not caption or not caption.strip():
        return []

    doc   = nlp(caption)
    seen  = set()
    ents  = []
    for ent in doc.ents:
        if ent.label_ in NER_TYPES:
            text = ent.text.strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                ents.append(text)

    # Fallback: noun chunks if fewer than 1 named entity
    if len(ents) < 1:
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                ents.append(text)

    return ents[:MAX_ENTITIES]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Wikipedia API
# ══════════════════════════════════════════════════════════════════════════════
def fetch_wiki_summary(entity: str) -> Optional[str]:
    """Fetch Wikipedia intro for entity. Returns None if not found."""
    params = {
        "action":       "query",
        "titles":       entity,
        "prop":         "extracts",
        "exintro":      True,
        "explaintext":  True,
        "redirects":    True,
        "format":       "json",
    }
    for attempt in range(1, WIKI_RETRIES + 2):   # 1 try + WIKI_RETRIES retries
        try:
            resp = requests.get(WIKI_API, params=params,
                                headers=WIKI_HEADERS, timeout=WIKI_TIMEOUT)
            resp.raise_for_status()
            data  = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id == "-1":
                    return None                    # page not found
                extract = page.get("extract", "").strip()
                if extract:
                    return extract[:WIKI_MAX_CHARS]
            return None
        except requests.RequestException as exc:
            if attempt <= WIKI_RETRIES:
                log.debug(f"Wikipedia retry {attempt} for '{entity}': {exc}")
                time.sleep(0.5 * attempt)
            else:
                log.debug(f"Wikipedia failed for '{entity}': {exc}")
                return None
    return None


def build_wiki_context(entities: list[str]) -> tuple[str, int]:
    """
    Query Wikipedia for each entity (with ENTITY_DELAY between queries).
    Returns (combined_context, context_length).
    """
    parts = []
    for i, ent in enumerate(entities):
        if i > 0:
            time.sleep(ENTITY_DELAY)
        summary = fetch_wiki_summary(ent)
        if summary:
            parts.append(f"[{ent}] {summary}")

    context = " ".join(parts)
    return context, len(context)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DeBERTa NLI Scoring
# ══════════════════════════════════════════════════════════════════════════════
class NLIScorer:
    NEUTRAL_RESULT = {
        "entailment_score":    0.5,
        "contradiction_score": 0.5,
        "neutral_score":       0.0,
        "factcheck_score":     0.0,
    }

    def __init__(self, model_id: str = NLI_MODEL_ID, device: int = 0):
        dev_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading NLI model on {dev_str} ...")
        self.device    = torch.device(dev_str)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
                             model_id).to(self.device)
        self.model.eval()

        # Detect label order from config
        id2label = self.model.config.id2label          # e.g. {0:'contradiction', 1:'entailment', 2:'neutral'}
        self.label2idx = {v.lower(): k for k, v in id2label.items()}
        self.con_idx   = self.label2idx.get("contradiction", 0)
        self.ent_idx   = self.label2idx.get("entailment",    1)
        self.neu_idx   = self.label2idx.get("neutral",       2)
        log.info(f"NLI labels: {id2label}")

    def score(self, caption: str, wiki_context: str) -> dict:
        """Score caption against wiki_context. Returns dict of scores."""
        if not wiki_context.strip():
            return self.NEUTRAL_RESULT.copy()

        inputs = self.tokenizer(
            caption,
            wiki_context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits        # shape: (1, 3)
        probs = F.softmax(logits, dim=-1).cpu().squeeze().tolist()

        ent = probs[self.ent_idx]
        con = probs[self.con_idx]
        neu = probs[self.neu_idx]
        return {
            "entailment_score":    round(ent, 6),
            "contradiction_score": round(con, 6),
            "neutral_score":       round(neu, 6),
            "factcheck_score":     round(ent - con, 6),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Full per-caption pipeline
# ══════════════════════════════════════════════════════════════════════════════
def factcheck_caption(caption: str, nlp, scorer: NLIScorer) -> dict:
    """
    Run full pipeline for one caption.
    Returns result dict including scores and diagnostic fields.
    """
    caption = (caption or "").strip()

    if not caption:
        return {**NLIScorer.NEUTRAL_RESULT, "entities": [],
                "wiki_context_length": 0, "status": "empty_caption"}

    entities = extract_entities(caption, nlp)
    if not entities:
        return {**NLIScorer.NEUTRAL_RESULT, "entities": [],
                "wiki_context_length": 0, "status": "no_entities"}

    wiki_context, ctx_len = build_wiki_context(entities)
    if ctx_len == 0:
        return {**NLIScorer.NEUTRAL_RESULT, "entities": entities,
                "wiki_context_length": 0, "status": "no_wiki_result"}

    scores = scorer.score(caption, wiki_context)
    return {**scores, "entities": entities,
            "wiki_context_length": ctx_len, "status": "ok"}


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MMFakeBench Dataset Loader
# ══════════════════════════════════════════════════════════════════════════════
def find_val_json() -> Optional[Path]:
    for p in VAL_SEARCH_PATHS:
        if p.exists():
            return p
    return None


def detect_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    low = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def load_mmfakebench_val() -> pd.DataFrame:
    path = find_val_json()
    if path is None:
        raise FileNotFoundError(
            f"Could not find val JSON. Searched:\n" +
            "\n".join(f"  {p}" for p in VAL_SEARCH_PATHS)
        )
    log.info(f"Loading val set from: {path}")

    if path.suffix == ".jsonl":
        with open(path) as f:
            data = [json.loads(l) for l in f if l.strip()]
        df = pd.DataFrame(data)
    else:
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict) and "annotations" in raw:
            df = pd.DataFrame(raw["annotations"])
        else:
            df = pd.DataFrame(list(raw.values()) if raw else [])

    log.info(f"Loaded {len(df)} rows.  Columns: {df.columns.tolist()}")

    cap_col = detect_column(df.columns.tolist(), CAPTION_COLS)
    lbl_col = detect_column(df.columns.tolist(), LABEL_COLS)
    log.info(f"Auto-detected caption='{cap_col}'  label='{lbl_col}'")

    if cap_col is None:
        raise ValueError(f"No caption column found. Available: {df.columns.tolist()}")

    df = df.rename(columns={cap_col: "__caption__"})
    if lbl_col:
        df = df.rename(columns={lbl_col: "__label__"})
    else:
        df["__label__"] = None
        log.warning("No label column detected — evaluation metrics will be skipped.")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Batch Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def run_evaluation(nlp, scorer: NLIScorer,
                   out_csv: Path = OUT_CSV,
                   ckpt_csv: Path = CKPT_CSV) -> pd.DataFrame:

    df = load_mmfakebench_val()
    records = []

    log.info(f"Running fact-check pipeline on {len(df)} captions ...")

    for idx, row in df.iterrows():
        caption = str(row.get("__caption__", "") or "")
        label   = row.get("__label__", None)

        result  = factcheck_caption(caption, nlp, scorer)

        records.append({
            "idx":                  idx,
            "caption":              caption,
            "label":                label,
            "entities":             "|".join(str(e) for e in result["entities"]),
            "factcheck_score":      result["factcheck_score"],
            "entailment_score":     result["entailment_score"],
            "contradiction_score":  result["contradiction_score"],
            "neutral_score":        result["neutral_score"],
            "wiki_context_length":  result["wiki_context_length"],
            "status":               result["status"],
        })

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(records).to_csv(ckpt_csv, index=False)
            log.info(f"  Checkpoint saved at sample {idx + 1}")

    results_df = pd.DataFrame(records)
    results_df.to_csv(out_csv, index=False)
    log.info(f"Final scores saved to {out_csv}")
    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Summary Report
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(df: pd.DataFrame) -> None:
    total   = len(df)
    ok_mask = df["status"] == "ok"
    scored  = ok_mask.sum()

    print("\n" + "="*60)
    print("  WIKIPEDIA FACT-CHECK — EVALUATION SUMMARY")
    print("="*60)
    print(f"  Total samples     : {total}")
    print(f"  Successfully scored: {scored}  ({scored/total*100:.1f}%)")
    print()

    print("  Status breakdown:")
    for status, count in df["status"].value_counts().items():
        print(f"    {status:<22}: {count:>4}  ({count/total*100:.1f}%)")
    print()

    scores = df["factcheck_score"]
    print("  factcheck_score stats (all samples):")
    print(f"    mean={scores.mean():.4f}  std={scores.std():.4f}  "
          f"min={scores.min():.4f}  max={scores.max():.4f}")
    print()

    # Per-class breakdown if labels exist
    if "__label__" in df.columns and df["__label__"].notna().any():
        # Normalise labels → 0=real, 1=fake
        lbl = df["__label__"].astype(str).str.strip().str.lower()
        real_mask = lbl.isin({"true", "real", "0", "false"})
        fake_mask = lbl.isin({"fake", "false", "1", "true"}) & ~real_mask

        # Handle gt_answers: 'True' = real, 'Fake' = fake
        raw = df["__label__"].astype(str).str.strip()
        real_mask = raw.isin({"True", "true", "real", "0"})
        fake_mask = raw.isin({"Fake", "fake", "false", "1"})

        print("  Mean factcheck_score by class:")
        if real_mask.any():
            print(f"    Real  (n={real_mask.sum()}): {df.loc[real_mask,'factcheck_score'].mean():.4f}")
        if fake_mask.any():
            print(f"    Fake  (n={fake_mask.sum()}): {df.loc[fake_mask,'factcheck_score'].mean():.4f}")
        print()

        # Threshold accuracy: score < 0.0 → predicted fake
        if real_mask.any() or fake_mask.any():
            known_mask = real_mask | fake_mask
            y_true = fake_mask[known_mask].astype(int)
            y_pred = (df.loc[known_mask, "factcheck_score"] < 0.0).astype(int)
            acc = (y_true.values == y_pred.values).mean()
            print(f"  Threshold accuracy (score < 0 => fake): {acc*100:.2f}%")
    print("="*60)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Quick Demo
# ══════════════════════════════════════════════════════════════════════════════
DEMO_CAPTIONS = [
    # Clearly real — verifiable Wikipedia fact
    ("Barack Obama was the 44th President of the United States, serving from 2009 to 2017.",
     "REAL (verifiable fact)"),
    # Clearly fake — contradicts Wikipedia
    ("The Eiffel Tower is located in Berlin, Germany, and was built by Adolf Hitler in 1942.",
     "FAKE (multiple contradictions)"),
    # Ambiguous — generic, hard to verify from Wikipedia
    ("A man was seen near a local market during the afternoon.",
     "AMBIGUOUS (no specific entities)"),
]

def run_demo(nlp, scorer: NLIScorer) -> None:
    print("\n" + "="*60)
    print("  QUICK DEMO — 3 CAPTIONS")
    print("="*60)
    for caption, note in DEMO_CAPTIONS:
        result = factcheck_caption(caption, nlp, scorer)
        print(f"\n  Caption : {caption}")
        print(f"  Note    : {note}")
        print(f"  Entities: {result['entities']}")
        print(f"  Status  : {result['status']}")
        print(f"  Entailment   : {result['entailment_score']:.4f}")
        print(f"  Contradiction: {result['contradiction_score']:.4f}")
        print(f"  Neutral      : {result['neutral_score']:.4f}")
        print(f"  factcheck_score: {result['factcheck_score']:+.4f}"
              f"  ({'supports' if result['factcheck_score'] > 0.1 else 'contradicts' if result['factcheck_score'] < -0.1 else 'neutral'})")
    print("="*60)


# ══════════════════════════════════════════════════════════════════════════════
# 10. Entry Point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Load models
    nlp    = load_spacy()
    scorer = NLIScorer(model_id=NLI_MODEL_ID, device=0)

    # Demo (runs first — fast sanity check before the long eval)
    run_demo(nlp, scorer)

    # Full MMFakeBench evaluation
    results_df = run_evaluation(nlp, scorer, out_csv=OUT_CSV, ckpt_csv=CKPT_CSV)

    # Summary report
    print_summary(results_df)
