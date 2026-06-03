"""
Wikipedia Fact-Check — NewsClipPings Spot-Check
================================================
Runs the Wikipedia fact-checking pipeline on 200 NewsClipPings val samples
(100 real, 100 fake) and compares coverage + signal separation against
the MMFakeBench results reported in mmfakebench_factcheck_scores.csv.

Imports reused directly from wikipedia_factcheck.py (no rewrites):
  load_spacy, NLIScorer, extract_entities, fetch_wiki_summary,
  build_wiki_context, factcheck_caption
"""
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402


import json, random
import numpy as np
import pandas as pd
from pathlib import Path

from wikipedia_factcheck import (
    load_spacy,
    NLIScorer,
    NLI_MODEL_ID,
    factcheck_caption,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(str(_cfg.ROOT))
VAL_LABELS    = PROJECT_ROOT / "dataset" / "data" / "NewsClipPings" / "merged_balanced" / "val.json"
VAL_META      = PROJECT_ROOT / "dataset" / "data" / "NewsClipPings" / "metadata" / "val.json"
OUT_CSV       = PROJECT_ROOT / "newsclip_factcheck_spotcheck.csv"

# MMFakeBench reference numbers (from completed evaluation)
MMFAKE_TOTAL        = 1000
MMFAKE_OK_PCT       = 54.0
MMFAKE_NOWIKI_PCT   = 46.0
MMFAKE_REAL_MEAN    = 0.0053
MMFAKE_FAKE_MEAN    = -0.0200
MMFAKE_SEPARATION   = abs(MMFAKE_REAL_MEAN - MMFAKE_FAKE_MEAN)
MMFAKE_THRESH_ACC   = 33.4

SAMPLE_SIZE = 100   # per class
SEED        = 42


# ── Data loading ───────────────────────────────────────────────────────────────
def load_newsclip_sample() -> pd.DataFrame:
    with open(VAL_LABELS) as f:
        raw = json.load(f)
    annotations = raw["annotations"]

    with open(VAL_META) as f:
        meta = json.load(f)   # keyed by str(image_id)

    real = [a for a in annotations if not a["falsified"]]
    fake = [a for a in annotations if     a["falsified"]]

    rng = random.Random(SEED)
    rng.shuffle(real)
    rng.shuffle(fake)
    sampled = real[:SAMPLE_SIZE] + fake[:SAMPLE_SIZE]

    rows = []
    for ann in sampled:
        m       = meta.get(str(ann["image_id"]), {})
        caption = (m.get("caption") or "").strip()
        rows.append({
            "image_id": ann["image_id"],
            "caption":  caption,
            "label":    1 if ann["falsified"] else 0,   # 1=fake, 0=real
            "source":   m.get("source", ""),
        })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} samples  "
          f"(real={( df['label']==0).sum()}, fake={(df['label']==1).sum()})")
    return df


# ── Evaluation ─────────────────────────────────────────────────────────────────
def run_spotcheck(nlp, scorer: NLIScorer) -> pd.DataFrame:
    df = load_newsclip_sample()
    records = []

    for i, row in df.iterrows():
        result = factcheck_caption(row["caption"], nlp, scorer)
        records.append({
            "idx":                  i,
            "caption":              row["caption"],
            "label":                row["label"],
            "entities":             "|".join(str(e) for e in result["entities"]),
            "factcheck_score":      result["factcheck_score"],
            "entailment_score":     result["entailment_score"],
            "contradiction_score":  result["contradiction_score"],
            "neutral_score":        result["neutral_score"],
            "wiki_context_length":  result["wiki_context_length"],
            "status":               result["status"],
        })

        done = len(records)
        if done % 20 == 0:
            print(f"  {done}/{len(df)} processed ...")

    out = pd.DataFrame(records)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV}")
    return out


# ── Comparison summary ─────────────────────────────────────────────────────────
def print_comparison(df: pd.DataFrame) -> None:
    total   = len(df)
    ok_mask = df["status"] == "ok"
    ok_n    = ok_mask.sum()
    nw_n    = (df["status"] == "no_wiki_result").sum()
    ok_pct  = ok_n  / total * 100
    nw_pct  = nw_n  / total * 100

    # Scored-only per-class means
    scored      = df[ok_mask]
    real_scored = scored[scored["label"] == 0]["factcheck_score"]
    fake_scored = scored[scored["label"] == 1]["factcheck_score"]
    real_mean   = real_scored.mean() if len(real_scored) else float("nan")
    fake_mean   = fake_scored.mean() if len(fake_scored) else float("nan")
    separation  = abs(real_mean - fake_mean)

    # Threshold accuracy across ALL samples (unscored default to 0.0)
    y_true      = df["label"].values
    y_pred      = (df["factcheck_score"] < 0.0).astype(int).values
    thresh_acc  = (y_true == y_pred).mean() * 100

    # Sign convention for display
    def fmt(v):
        return f"{v:+.4f}" if not np.isnan(v) else "  N/A "

    print()
    print("=" * 60)
    print("NEWSCLIP SPOT-CHECK vs MMFAKEBENCH COMPARISON")
    print("=" * 60)
    print(f"\n{'':26} {'NewsClip':>10}  {'MMFakeBench':>12}")
    print(f"  {'Total samples':<24} {total:>10}  {MMFAKE_TOTAL:>12}")
    print(f"  {'Coverage (ok%)':<24} {ok_pct:>9.1f}%  {MMFAKE_OK_PCT:>11.1f}%")
    print(f"  {'No wiki result%':<24} {nw_pct:>9.1f}%  {MMFAKE_NOWIKI_PCT:>11.1f}%")

    print(f"\n  Factcheck Score (scored-only):")
    print(f"  {'Real mean':<24} {fmt(real_mean):>10}  {MMFAKE_REAL_MEAN:>+12.4f}")
    print(f"  {'Fake mean':<24} {fmt(fake_mean):>10}  {MMFAKE_FAKE_MEAN:>+12.4f}")
    print(f"  {'Separation (delta mean)':<24} {separation:>10.4f}  {MMFAKE_SEPARATION:>12.4f}")

    print(f"\n  Threshold accuracy (< 0.0 -> fake):")
    print(f"  {'NewsClip':<24} {thresh_acc:>9.2f}%")
    print(f"  {'MMFakeBench':<24} {MMFAKE_THRESH_ACC:>9.1f}%")

    print()
    print("=" * 60)
    print("INTERPRETATION")

    better_coverage   = ok_pct   > MMFAKE_OK_PCT
    better_separation = separation > MMFAKE_SEPARATION

    if better_coverage and better_separation:
        print("  Coverage > 54% and separation > 0.015:")
        print("  -> Wikipedia module adds value on NewsClipPings")
        print("  -> Consider adding as conditional signal in NewsClip fusion")
    else:
        findings = []
        if not better_coverage:
            findings.append(f"coverage {ok_pct:.1f}% <= MMFakeBench 54.0%")
        if not better_separation:
            findings.append(f"separation {separation:.4f} <= MMFakeBench 0.0147")
        print(f"  Similar to MMFakeBench ({', '.join(findings)}):")
        print("  -> Module is dataset-agnostic weak signal, thesis finding confirmed")
    print("=" * 60)

    # Extra breakdown
    print(f"\n  Status breakdown (NewsClip):")
    for status, count in df["status"].value_counts().items():
        print(f"    {status:<22}: {count:>3}  ({count/total*100:.1f}%)")

    print(f"\n  Scored samples by class:")
    print(f"    Real scored : {len(real_scored)}/{(df['label']==0).sum()}")
    print(f"    Fake scored : {len(fake_scored)}/{(df['label']==1).sum()}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    nlp    = load_spacy()
    scorer = NLIScorer(model_id=NLI_MODEL_ID, device=0)

    results_df = run_spotcheck(nlp, scorer)
    print_comparison(results_df)
