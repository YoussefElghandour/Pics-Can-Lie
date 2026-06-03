"""
2.3 — compute_deberta_nli_mmfb.py
---------------------------------
For each MMFakeBench caption (train + val), build a Wikipedia-derived
"premise" from the top-N entity summaries and score the caption (hypothesis)
against it with cross-encoder/nli-deberta-v3-large.

Output schema (per split CSV):
  sample_id, deberta_score, entity_count, has_evidence

  deberta_score = softmax(logits)[entailment_idx]
                = neutral 0.5 if has_evidence is False (no entities OR all
                  entities miss / cache-empty).
  has_evidence  = 1 iff the assembled premise is non-empty
  entity_count  = entities found by 2.1 for this caption

CRITICAL — the NLIScorer auto-detects the label index from
model.config.id2label (handles the [contradiction, neutral, entailment]
ordering question robustly). We additionally run a known-good sanity pair
("The sky is blue." entails itself) and abort if entailment < 0.85 — that
would mean the index resolved to the wrong column.

Resume-safe: writes a checkpoint CSV every CHECKPOINT_EVERY rows and
skips already-scored sample_ids on restart.

Run:
    python mmfakebench_training/compute_deberta_nli_mmfb.py --split both
"""


from __future__ import annotations
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wikipedia_factcheck import NLI_MODEL_ID, WIKI_MAX_CHARS  # noqa: E402

PROJECT_ROOT = Path(str(_cfg.ROOT))
MMFB_ROOT    = Path(_os.path.join(str(_cfg.MMFB_ROOT)))
OUT_ROOT     = PROJECT_ROOT / "mmfakebench_training"
WIKI_CACHE   = PROJECT_ROOT / "wiki_page_cache.json"

SPLITS = {
    "train": {
        "json":     MMFB_ROOT / "MMFakeBench_test.json",
        "entities": OUT_ROOT / "caption_entities_train.json",
        "out":      OUT_ROOT / "deberta_nli_train.csv",
        "ckpt":     OUT_ROOT / "_deberta_ckpt_train.csv",
    },
    "val": {
        "json":     MMFB_ROOT / "MMFakeBench_val.json",
        "entities": OUT_ROOT / "caption_entities_val.json",
        "out":      OUT_ROOT / "deberta_nli_val.csv",
        "ckpt":     OUT_ROOT / "_deberta_ckpt_val.csv",
    },
}

TOP_K_ENTITIES   = 3       # use top-3 entity summaries as premise
PREMISE_MAX_CHARS = 1500   # truncate combined premise (DeBERTa max 512 tokens)
BATCH_SIZE       = 8       # 4060 8GB cap for DeBERTa-large fp32
CHECKPOINT_EVERY = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Tokenizer / model load (fp32; DeBERTa-v3-large needs fp32 layers) ───────
def load_nli():
    print(f"[nli] loading {NLI_MODEL_ID} on {DEVICE} (fp32)")
    tok = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID)
    model = model.float().to(DEVICE)   # NOT .half()
    model.eval()
    id2label = model.config.id2label
    print(f"[nli] id2label = {id2label}")
    label2idx = {v.lower(): int(k) for k, v in id2label.items()}
    ent_idx = label2idx["entailment"]
    print(f"[nli] entailment_idx = {ent_idx}")
    return tok, model, ent_idx


@torch.no_grad()
def sanity_check(tok, model, ent_idx: int) -> None:
    inputs = tok(
        ["The sky is blue."], ["The sky is blue."],
        return_tensors="pt", truncation=True, padding=True,
    ).to(DEVICE)
    probs = F.softmax(model(**inputs).logits, dim=-1).cpu().squeeze().tolist()
    print(f"[sanity] entailment-of-self probs = {[round(p,4) for p in probs]} "
          f"(entailment column @ idx {ent_idx} = {probs[ent_idx]:.4f})")
    if probs[ent_idx] < 0.85:
        raise SystemExit(
            f"[sanity] FAIL: ent={probs[ent_idx]:.3f} < 0.85 — wrong index? "
            f"check NLIScorer label resolution."
        )
    print("[sanity] OK — entailment column resolves correctly.")


def build_premise(ents: List[str], cache: dict) -> tuple[str, bool]:
    """Return (premise_text, has_evidence). Uses top-K entities, dedup by key."""
    parts: List[str] = []
    seen: set[str] = set()
    for ent in ents[:TOP_K_ENTITIES]:
        for key in (ent.strip(), ent.strip().replace(" ", "_")):
            if key in seen:
                continue
            summary = cache.get(key)
            if summary:
                parts.append(f"[{ent}] {summary[:WIKI_MAX_CHARS]}")
                seen.add(key)
                break
    if not parts:
        return "", False
    premise = " ".join(parts)
    return premise[:PREMISE_MAX_CHARS], True


@torch.no_grad()
def score_batch(captions: List[str], premises: List[str], tok, model, ent_idx: int) -> np.ndarray:
    """Batch-score a list of (caption, premise) pairs. Returns entailment probs."""
    inputs = tok(
        captions, premises,
        return_tensors="pt", truncation=True, max_length=512, padding=True,
    ).to(DEVICE)
    logits = model(**inputs).logits           # (B, 3)
    probs = F.softmax(logits, dim=-1)
    return probs[:, ent_idx].cpu().numpy()


def run_split(name: str, tok, model, ent_idx: int, cache: dict) -> None:
    cfg = SPLITS[name]
    with open(cfg["json"], encoding="utf-8") as f:
        ann = json.load(f)
    with open(cfg["entities"], encoding="utf-8") as f:
        ents_by_id = json.load(f)

    n = len(ann)
    sample_ids = [f"{name}_{i:06d}" for i in range(n)]
    scores      = np.full(n, np.nan, dtype=np.float32)
    has_evid    = np.zeros(n, dtype=np.int8)
    ent_counts  = np.zeros(n, dtype=np.int16)

    # ── Resume ──────────────────────────────────────────────────────────────
    ckpt_csv: Path = cfg["ckpt"]
    if ckpt_csv.exists():
        prev = pd.read_csv(ckpt_csv).set_index("sample_id")
        mask = prev["deberta_score"].notna()
        done = prev.index[mask].tolist()
        sid_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
        for sid in done:
            i = sid_to_idx[sid]
            scores[i]     = float(prev.at[sid, "deberta_score"])
            has_evid[i]   = int(prev.at[sid, "has_evidence"])
            ent_counts[i] = int(prev.at[sid, "entity_count"])
        print(f"[{name}] resumed: {len(done)}/{n} already scored")

    # ── Build pending list of (idx, caption, ents) ──────────────────────────
    pending: list[tuple[int, str, list[str]]] = []
    for i, s in enumerate(ann):
        if not np.isnan(scores[i]):
            continue
        sid = sample_ids[i]
        cap = (s.get("text", "") or "").strip()
        ents = ents_by_id.get(sid, [])
        pending.append((i, cap, ents))
    print(f"[{name}] pending: {len(pending)} / {n}")

    # ── Batch score ─────────────────────────────────────────────────────────
    since_ckpt = 0
    pbar = tqdm(range(0, len(pending), BATCH_SIZE), desc=f"[{name}] DeBERTa", unit="batch")
    for s in pbar:
        batch = pending[s:s + BATCH_SIZE]
        idxs, caps, ents_list = [], [], []
        # Resolve premise per sample; samples with no evidence are scored with 0.5 directly
        for i, cap, ents in batch:
            premise, has_e = build_premise(ents, cache)
            ent_counts[i] = len(ents)
            has_evid[i] = int(has_e)
            if not has_e or not cap:
                scores[i] = 0.5
            else:
                idxs.append(i); caps.append(cap); ents_list.append(premise)

        if caps:
            ent_probs = score_batch(caps, ents_list, tok, model, ent_idx)
            for i, p in zip(idxs, ent_probs):
                scores[i] = float(p)

        since_ckpt += len(batch)
        if since_ckpt >= CHECKPOINT_EVERY:
            _write_csv(ckpt_csv, sample_ids, scores, ent_counts, has_evid)
            since_ckpt = 0

    # ── Final ───────────────────────────────────────────────────────────────
    _write_csv(cfg["out"], sample_ids, scores, ent_counts, has_evid)
    if ckpt_csv.exists():
        ckpt_csv.unlink()
    print(f"[{name}] saved -> {cfg['out']}  ({n} rows)")

    # ── Quick distribution sanity (per fake_cls) ────────────────────────────
    df = pd.DataFrame({
        "sample_id": sample_ids,
        "deberta_score": scores,
        "entity_count": ent_counts,
        "has_evidence": has_evid,
        "fake_cls":  [s.get("fake_cls", "")   for s in ann],
        "gt":        [s.get("gt_answers", "") for s in ann],
    })
    print(f"[{name}] has_evidence={int(has_evid.sum())}/{n} "
          f"({has_evid.sum()/n:.1%})")
    print(f"[{name}] deberta_score mean by fake_cls (has_evidence only):")
    for cat in ["original", "mismatch", "textual_veracity_distortion", "visual_veracity_distortion"]:
        m = (df["fake_cls"] == cat) & (df["has_evidence"] == 1)
        if m.any():
            print(f"    {cat:<35} n={int(m.sum()):<4} "
                  f"mean={df.loc[m,'deberta_score'].mean():.4f}  "
                  f"median={df.loc[m,'deberta_score'].median():.4f}")


def _write_csv(path: Path, sample_ids, scores, ent_counts, has_evid) -> None:
    df = pd.DataFrame({
        "sample_id": sample_ids,
        "deberta_score": scores,
        "entity_count": ent_counts,
        "has_evidence": has_evid,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    args = ap.parse_args()

    if not WIKI_CACHE.exists():
        sys.exit(f"[!] wiki cache missing at {WIKI_CACHE} — run 2.2 first")
    with open(WIKI_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    print(f"[cache] {len(cache)} wiki entries")

    tok, model, ent_idx = load_nli()
    sanity_check(tok, model, ent_idx)

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for s in splits:
        run_split(s, tok, model, ent_idx, cache)
    print("\nDone.")


if __name__ == "__main__":
    main()
