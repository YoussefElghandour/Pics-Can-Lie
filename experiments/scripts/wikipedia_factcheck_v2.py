"""
Wikipedia Fact-Checking Module — v2 (Retrieval-Augmented)
==========================================================
Design change vs v1:
  v1  – fetch only the Wikipedia *intro* (exintro=True, <=1500 chars).
  v2  – fetch the *full* article, chunk it into paragraphs, rerank the
        chunks by TF-IDF cosine similarity to the caption, then feed the
        top-k most relevant paragraphs to the DeBERTa NLI scorer.

All shared utilities (load_spacy, extract_entities, NLIScorer, detect_column,
load_mmfakebench_val, ...) are imported from wikipedia_factcheck.py -- not copied.

Resume logic: if a checkpoint CSV exists at startup, the already-processed
rows are loaded and the loop skips ahead to the first unprocessed sample.
"""

import sys
import time
import re
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Ensure the project root is importable ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Shared imports from v1 (no copy-paste) ────────────────────────────────────
from wikipedia_factcheck import (
    load_spacy,
    extract_entities,
    NLIScorer,
    detect_column,
    find_val_json,
    load_mmfakebench_val,
    # Constants
    WIKI_API,
    WIKI_HEADERS,
    WIKI_TIMEOUT,
    WIKI_RETRIES,
    ENTITY_DELAY,
    MAX_ENTITIES,
    NLI_MODEL_ID,
    NER_TYPES,
    CHECKPOINT_EVERY,
    PROJECT_ROOT,
    CAPTION_COLS,
    LABEL_COLS,
)

# ── Logging ────────────────────────────────────────────────────────────────────
log = logging.getLogger("factcheck_v2")

# ── v2-specific constants ─────────────────────────────────────────────────────
WIKI_FULL_CHARS   = 8_000   # max chars fetched from the full article
CHUNK_MIN_CHARS   = 80      # discard paragraphs shorter than this
RERANK_TOP_K      = 3       # top-k paragraphs selected per entity
RERANKED_MAX_CTX  = 1_500   # context chars fed to NLI (same cap as v1 for fair comparison)

OUT_CSV_V2  = PROJECT_ROOT / "mmfakebench_factcheck_scores_v2.csv"
CKPT_CSV_V2 = PROJECT_ROOT / "mmfakebench_factcheck_checkpoint_v2.csv"
OUT_CSV_V1  = PROJECT_ROOT / "mmfakebench_factcheck_scores.csv"   # for comparison

CIRCUIT_BREAKER_THRESHOLD = 15   # consecutive no_wiki_result before tripping
HEARTBEAT_EVERY            = 100  # check connectivity every N samples
CONNECTIVITY_RETRY_INTERVAL = 30  # seconds between retries
CONNECTIVITY_MAX_WAIT       = 300 # seconds (5 minutes) before giving up

_CONNECTIVITY_URL    = "https://en.wikipedia.org/w/api.php"
_CONNECTIVITY_PARAMS = {"action": "query", "format": "json", "titles": "Barack_Obama"}


# ══════════════════════════════════════════════════════════════════════════════
# 0.  Connectivity utilities
# ══════════════════════════════════════════════════════════════════════════════
def check_wikipedia_connectivity() -> bool:
    """Hit a known-good Wikipedia endpoint; return True iff status 200 and 'query' key present."""
    try:
        resp = requests.get(
            _CONNECTIVITY_URL,
            params=_CONNECTIVITY_PARAMS,
            headers=WIKI_HEADERS,
            timeout=10,
        )
        return resp.status_code == 200 and "query" in resp.json()
    except Exception:
        return False


def _wait_for_connectivity(source: str) -> bool:
    """
    Retry check_wikipedia_connectivity() every CONNECTIVITY_RETRY_INTERVAL seconds
    for up to CONNECTIVITY_MAX_WAIT seconds.  Returns True if restored, False on timeout.
    """
    elapsed = 0
    while elapsed < CONNECTIVITY_MAX_WAIT:
        time.sleep(CONNECTIVITY_RETRY_INTERVAL)
        elapsed += CONNECTIVITY_RETRY_INTERVAL
        log.info(f"[{source}] Retrying connectivity check ({elapsed}s / {CONNECTIVITY_MAX_WAIT}s) ...")
        if check_wikipedia_connectivity():
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Full-article fetch
# ══════════════════════════════════════════════════════════════════════════════
def fetch_wiki_full_article(entity: str) -> Optional[str]:
    """Fetch the full Wikipedia article (not just the intro). Returns None on miss."""
    params = {
        "action":      "query",
        "titles":      entity,
        "prop":        "extracts",
        "explaintext": True,
        "redirects":   True,
        "format":      "json",
        # no exintro=True -- we want the whole article
    }
    for attempt in range(1, WIKI_RETRIES + 2):
        try:
            resp = requests.get(WIKI_API, params=params,
                                headers=WIKI_HEADERS, timeout=WIKI_TIMEOUT)
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id == "-1":
                    return None
                text = page.get("extract", "").strip()
                return text[:WIKI_FULL_CHARS] if text else None
            return None
        except requests.RequestException as exc:
            if attempt <= WIKI_RETRIES:
                log.debug(f"Retry {attempt} for '{entity}': {exc}")
                time.sleep(0.5 * attempt)
            else:
                log.debug(f"Wikipedia failed for '{entity}': {exc}")
                return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Chunking
# ══════════════════════════════════════════════════════════════════════════════
_SECTION_HEADER_RE = re.compile(r"^={2,}\s*.+\s*={2,}$")


def chunk_into_paragraphs(text: str) -> list[str]:
    """
    Split article text on blank lines; drop section-header lines and
    paragraphs shorter than CHUNK_MIN_CHARS.
    """
    chunks = []
    for raw in re.split(r"\n{2,}", text):
        chunk = raw.strip()
        if not chunk:
            continue
        if _SECTION_HEADER_RE.match(chunk):   # e.g. "== History =="
            continue
        if len(chunk) < CHUNK_MIN_CHARS:
            continue
        chunks.append(chunk)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TF-IDF reranking
# ══════════════════════════════════════════════════════════════════════════════
def rerank_paragraphs(caption: str, paragraphs: list[str],
                      top_k: int = RERANK_TOP_K) -> tuple[list[str], float]:
    """
    Rank paragraphs by TF-IDF cosine similarity to caption.
    Returns (top_k paragraphs in original order, mean similarity of selected chunks).
    """
    if not paragraphs:
        return [], 0.0

    try:
        corpus = [caption] + paragraphs
        tfidf  = TfidfVectorizer(stop_words="english", min_df=1).fit_transform(corpus)
        sims   = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

        if len(paragraphs) <= top_k:
            return paragraphs, float(sims.mean())

        top_idx  = sorted(np.argsort(sims)[-top_k:])   # restore reading order
        top_sims = sims[top_idx]
        return [paragraphs[i] for i in top_idx], float(top_sims.mean())

    except Exception:
        return paragraphs[:top_k], 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Retrieval-augmented context builder
# ══════════════════════════════════════════════════════════════════════════════
def build_reranked_context(
    entities: list[str],
    caption: str,
) -> tuple[str, int, int, int, float]:
    """
    For each entity:
      1. Fetch full article.
      2. Chunk into paragraphs.
      3. Rerank by TF-IDF similarity to caption, keep top-k.
    Returns:
      (context_str, context_len, total_chunks, entities_with_context, avg_top_sim)
    """
    selected_parts: list[str] = []
    total_chunks = 0
    sim_scores:  list[float] = []

    for i, ent in enumerate(entities):
        if i > 0:
            time.sleep(ENTITY_DELAY)
        article = fetch_wiki_full_article(ent)
        if not article:
            continue
        paragraphs    = chunk_into_paragraphs(article)
        total_chunks += len(paragraphs)
        top_chunks, avg_sim = rerank_paragraphs(caption, paragraphs)
        if top_chunks:
            selected_parts.append(f"[{ent}] " + " ".join(top_chunks))
            sim_scores.append(avg_sim)

    context     = " ".join(selected_parts)[:RERANKED_MAX_CTX]
    avg_top_sim = float(np.mean(sim_scores)) if sim_scores else 0.0
    return context, len(context), total_chunks, len(selected_parts), avg_top_sim


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Per-caption v2 pipeline
# ══════════════════════════════════════════════════════════════════════════════
def factcheck_caption_v2(caption: str, nlp, scorer: NLIScorer) -> dict:
    """Retrieval-augmented pipeline for a single caption."""
    caption = (caption or "").strip()
    _empty  = {"n_chunks": 0, "top_chunks_used": 0, "avg_top_sim": 0.0}

    if not caption:
        return {**NLIScorer.NEUTRAL_RESULT, "entities": [],
                "wiki_context_length": 0, "status": "empty_caption", **_empty}

    entities = extract_entities(caption, nlp)
    if not entities:
        return {**NLIScorer.NEUTRAL_RESULT, "entities": [],
                "wiki_context_length": 0, "status": "no_entities", **_empty}

    context, ctx_len, n_chunks, ents_with_ctx, avg_top_sim = \
        build_reranked_context(entities, caption)

    if ctx_len == 0:
        return {**NLIScorer.NEUTRAL_RESULT, "entities": entities,
                "wiki_context_length": 0, "status": "no_wiki_result",
                "n_chunks": n_chunks, "top_chunks_used": 0, "avg_top_sim": 0.0}

    scores = scorer.score(caption, context)
    return {
        **scores,
        "entities":            entities,
        "wiki_context_length": ctx_len,
        "status":              "ok",
        "n_chunks":            n_chunks,
        "top_chunks_used":     ents_with_ctx,
        "avg_top_sim":         avg_top_sim,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Batch evaluation (with checkpoint resume)
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_mmfakebench_v2(
    nlp,
    scorer: NLIScorer,
    max_samples: Optional[int] = None,
    out_csv: Path = OUT_CSV_V2,
    ckpt_csv: Path = CKPT_CSV_V2,
) -> pd.DataFrame:

    # ── Pre-flight connectivity check ─────────────────────────────────────
    log.info("[v2] Checking Wikipedia connectivity before starting ...")
    if not check_wikipedia_connectivity():
        log.error(
            "[v2] Wikipedia is unreachable. "
            "Fix network connectivity and restart the evaluation."
        )
        sys.exit(1)
    log.info("[v2] Wikipedia connectivity confirmed.")

    df = load_mmfakebench_val()
    if max_samples is not None:
        df = df.iloc[:max_samples]

    # ── Resume from checkpoint if it exists ───────────────────────────────
    records: list[dict] = []
    n_done  = 0
    if ckpt_csv.exists():
        ckpt_df = pd.read_csv(ckpt_csv)
        # Back-compat: older checkpoints may lack avg_top_sim
        if "avg_top_sim" not in ckpt_df.columns:
            ckpt_df["avg_top_sim"] = 0.0
        records = ckpt_df.to_dict(orient="records")
        n_done  = len(records)
        log.info(f"[v2] Checkpoint found -- resuming from sample {n_done} / {len(df)}")
    else:
        log.info("[v2] No checkpoint found -- starting fresh.")

    if n_done >= len(df):
        log.info("[v2] All samples already in checkpoint -- writing final CSV.")
        results_df = pd.DataFrame(records)
        results_df.to_csv(out_csv, index=False)
        return results_df

    log.info(f"[v2] Processing {len(df) - n_done} remaining captions ...")

    consecutive_no_wiki = 0

    for pos, (_, row) in enumerate(df.iterrows()):
        if pos < n_done:
            continue                        # already done -- fast-skip

        caption = str(row.get("__caption__", "") or "")
        label   = row.get("__label__", None)
        result  = factcheck_caption_v2(caption, nlp, scorer)

        records.append({
            "idx":                  pos,
            "caption":              caption,
            "label":                label,
            "entities":             "|".join(str(e) for e in result["entities"]),
            "factcheck_score":      result["factcheck_score"],
            "entailment_score":     result["entailment_score"],
            "contradiction_score":  result["contradiction_score"],
            "neutral_score":        result["neutral_score"],
            "wiki_context_length":  result["wiki_context_length"],
            "status":               result["status"],
            "n_chunks":             result["n_chunks"],
            "top_chunks_used":      result["top_chunks_used"],
            "avg_top_sim":          result["avg_top_sim"],
        })

        # ── Circuit breaker ────────────────────────────────────────────────
        if result["status"] == "no_wiki_result":
            consecutive_no_wiki += 1
        else:
            consecutive_no_wiki = 0

        if consecutive_no_wiki >= CIRCUIT_BREAKER_THRESHOLD:
            log.warning(
                f"CIRCUIT BREAKER TRIPPED — {CIRCUIT_BREAKER_THRESHOLD} consecutive "
                "no_wiki_result outcomes. Pausing to check connectivity."
            )
            if check_wikipedia_connectivity():
                log.info("Connectivity restored, resuming evaluation.")
            else:
                if not _wait_for_connectivity("circuit_breaker"):
                    pd.DataFrame(records).to_csv(ckpt_csv, index=False)
                    log.error(
                        f"[v2] No connectivity after {CONNECTIVITY_MAX_WAIT}s. "
                        f"Checkpoint saved to {ckpt_csv}. Exiting."
                    )
                    sys.exit(1)
                log.info("Connectivity restored, resuming evaluation.")
            consecutive_no_wiki = 0

        # ── Periodic heartbeat every HEARTBEAT_EVERY samples ──────────────
        samples_this_run = len(records) - n_done
        if samples_this_run > 0 and samples_this_run % HEARTBEAT_EVERY == 0:
            if not check_wikipedia_connectivity():
                log.warning(
                    f"[v2] Heartbeat FAILED at sample {len(records)} total "
                    f"({samples_this_run} this run). Pausing to check connectivity."
                )
                if not _wait_for_connectivity("heartbeat"):
                    pd.DataFrame(records).to_csv(ckpt_csv, index=False)
                    log.error(
                        f"[v2] No connectivity after {CONNECTIVITY_MAX_WAIT}s. "
                        f"Checkpoint saved to {ckpt_csv}. Exiting."
                    )
                    sys.exit(1)
                log.info("Connectivity restored after heartbeat failure, resuming evaluation.")

        if len(records) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(records).to_csv(ckpt_csv, index=False)
            log.info(f"  [v2] Checkpoint saved at sample {len(records)}")

    results_df = pd.DataFrame(records)
    results_df.to_csv(out_csv, index=False)
    log.info(f"[v2] Final scores saved -> {out_csv}")
    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Comparison summary (extended)
# ══════════════════════════════════════════════════════════════════════════════
def _stats(df: pd.DataFrame) -> dict:
    """Compute summary stats for one results DataFrame."""
    total = len(df)
    ok    = (df["status"] == "ok").sum()
    sc    = df["factcheck_score"]

    lbl_col = ("__label__" if "__label__" in df.columns
               else ("label" if "label" in df.columns else None))
    raw = df[lbl_col].astype(str).str.strip() if lbl_col else pd.Series([""] * total)

    real_mask = raw.isin({"True", "true", "real", "0"})
    fake_mask = raw.isin({"Fake", "fake", "false", "1"})
    known     = real_mask | fake_mask

    acc = None
    if known.any():
        y_true = fake_mask[known].astype(int)
        y_pred = (df.loc[known, "factcheck_score"] < 0.0).astype(int)
        acc    = (y_true.values == y_pred.values).mean()

    return {
        "n":            total,
        "coverage_pct": ok / total * 100,
        "mean":         sc.mean(),
        "std":          sc.std(),
        "min":          sc.min(),
        "max":          sc.max(),
        "mean_real":    df.loc[real_mask, "factcheck_score"].mean() if real_mask.any() else float("nan"),
        "mean_fake":    df.loc[fake_mask, "factcheck_score"].mean() if fake_mask.any() else float("nan"),
        "n_real":       int(real_mask.sum()),
        "n_fake":       int(fake_mask.sum()),
        "thresh_acc":   acc,
    }


def print_comparison_summary(df_v1: pd.DataFrame, df_v2: pd.DataFrame) -> None:
    s1 = _stats(df_v1)
    s2 = _stats(df_v2)

    W = 72

    # ── Main comparison table ─────────────────────────────────────────────
    print()
    print("=" * W)
    print("  WIKIPEDIA FACT-CHECK - V1 (intro) vs V2 (RAG) COMPARISON")
    print("=" * W)
    print(f"  {'Metric':<30} {'v1  intro-only':>15} {'v2  RAG':>15}  {'delta':>8}")
    print("-" * W)

    def _fmt(v, pct=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{v:.1f}%" if pct else f"{v:.4f}"

    def _delta(v1, v2, pct=False):
        if v1 is None or v2 is None:
            return ""
        try:
            d    = v2 - v1
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.1f}%" if pct else f"{sign}{d:.4f}"
        except Exception:
            return ""

    rows = [
        ("Samples (N)",          s1["n"],             s2["n"],             False),
        ("Coverage (% ok)",      s1["coverage_pct"],  s2["coverage_pct"],  True),
        ("Mean factcheck_score", s1["mean"],          s2["mean"],          False),
        ("Std  factcheck_score", s1["std"],           s2["std"],           False),
        ("Min  factcheck_score", s1["min"],           s2["min"],           False),
        ("Max  factcheck_score", s1["max"],           s2["max"],           False),
    ]
    for label, v1, v2, pct in rows:
        v1s = str(int(v1)) if label == "Samples (N)" else _fmt(v1, pct)
        v2s = str(int(v2)) if label == "Samples (N)" else _fmt(v2, pct)
        dl  = "" if label == "Samples (N)" else _delta(v1, v2, pct)
        print(f"  {label:<30} {v1s:>15} {v2s:>15}  {dl:>8}")

    print("-" * W)

    # Per-class breakdown
    print(f"  {'Mean score - REAL (n='+str(s1['n_real'])+')':<30} "
          f"{_fmt(s1['mean_real']):>15} {_fmt(s2['mean_real']):>15}"
          f"  {_delta(s1['mean_real'], s2['mean_real']):>8}")
    print(f"  {'Mean score - FAKE (n='+str(s1['n_fake'])+')':<30} "
          f"{_fmt(s1['mean_fake']):>15} {_fmt(s2['mean_fake']):>15}"
          f"  {_delta(s1['mean_fake'], s2['mean_fake']):>8}")

    print("-" * W)

    # Threshold accuracy
    ta1 = s1["thresh_acc"]
    ta2 = s2["thresh_acc"]
    ta1_s = _fmt(ta1 * 100, pct=True) if ta1 is not None else "n/a"
    ta2_s = _fmt(ta2 * 100, pct=True) if ta2 is not None else "n/a"
    dl    = _delta(ta1 * 100 if ta1 else None, ta2 * 100 if ta2 else None, pct=True)
    print(f"  {'Threshold acc (score < 0)':<30} {ta1_s:>15} {ta2_s:>15}  {dl:>8}")

    print("=" * W)

    # ── v2 status distribution ────────────────────────────────────────────
    print()
    print("  v2 status distribution:")
    total_v2 = len(df_v2)
    for status, count in df_v2["status"].value_counts().items():
        bar = "#" * int(count / total_v2 * 30)
        print(f"    {status:<22}: {count:>4}  ({count/total_v2*100:5.1f}%)  {bar}")

    # ── v2 retrieval stats ────────────────────────────────────────────────
    if "n_chunks" in df_v2.columns:
        ok_v2 = df_v2[df_v2["status"] == "ok"]
        if not ok_v2.empty:
            print()
            print("  v2 retrieval stats (ok-scored samples only):")
            print(f"    Avg paragraphs found per caption : {ok_v2['n_chunks'].mean():.1f}")
            if "avg_top_sim" in ok_v2.columns:
                print(f"    Avg top-k TF-IDF similarity      : {ok_v2['avg_top_sim'].mean():.4f}")
            print(f"    Avg context length fed to NLI    : {ok_v2['wiki_context_length'].mean():.0f} chars")
            print(f"    Median context length            : {ok_v2['wiki_context_length'].median():.0f} chars")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Wikipedia fact-check v2 (retrieval-augmented)"
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Evaluate only the first N samples (default: all 1000)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip evaluation; print comparison from existing CSVs")
    args = parser.parse_args()

    if args.summary_only:
        if not OUT_CSV_V1.exists():
            print(f"ERROR: v1 scores not found at {OUT_CSV_V1}")
            sys.exit(1)
        if not OUT_CSV_V2.exists():
            print(f"ERROR: v2 scores not found at {OUT_CSV_V2}")
            sys.exit(1)
        df_v1 = pd.read_csv(OUT_CSV_V1)
        df_v2 = pd.read_csv(OUT_CSV_V2)
        n = min(len(df_v1), len(df_v2))
        print_comparison_summary(df_v1.head(n), df_v2.head(n))
        sys.exit(0)

    # ── Full / partial evaluation run ─────────────────────────────────────
    nlp    = load_spacy()
    scorer = NLIScorer(model_id=NLI_MODEL_ID, device=0)

    df_v2 = evaluate_mmfakebench_v2(
        nlp, scorer,
        max_samples=args.max_samples,
        out_csv=OUT_CSV_V2,
        ckpt_csv=CKPT_CSV_V2,
    )

    print("\n-- v2 output (first 5 rows) --")
    cols = ["idx", "caption", "label", "factcheck_score",
            "status", "n_chunks", "avg_top_sim"]
    print(df_v2[cols].head().to_string(index=False))

    if OUT_CSV_V1.exists():
        df_v1 = pd.read_csv(OUT_CSV_V1)
        n = min(len(df_v1), len(df_v2))
        print_comparison_summary(df_v1.head(n), df_v2.head(n))
    else:
        log.warning(f"v1 CSV not found at {OUT_CSV_V1} -- skipping comparison.")
        from wikipedia_factcheck import print_summary
        print_summary(df_v2.rename(columns={"label": "__label__"}))
