"""
Colab-aware entry point for the Stage 1 Wikipedia fact-check evaluation.

Replaces direct invocation of wikipedia_factcheck_v2.py in Colab because
that script resolves paths from PROJECT_ROOT = Path(r"D:\\Pics Can Lie"),
which doesn't exist on a Linux Colab instance.

This wrapper applies path_overrides first, then delegates to the same
evaluation and summary logic that wikipedia_factcheck_v2.py uses.

Usage:
    python run_colab.py                   # resume from checkpoint (or fresh start)
    python run_colab.py --summary-only    # print V1 vs V2 comparison table
    python run_colab.py --max-samples 50  # quick smoke-test with 50 samples
"""

import sys
import logging
import argparse
from pathlib import Path

# ── Apply path overrides BEFORE importing evaluation modules ─────────────────
import path_overrides
from path_overrides import OUT_CSV_V2, CKPT_CSV_V2, OUT_CSV_V1

import pandas as pd
import wikipedia_factcheck_v2 as wf2
from wikipedia_factcheck import load_spacy, NLIScorer, NLI_MODEL_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_colab")

parser = argparse.ArgumentParser(description="Stage 1 Wikipedia fact-check (Colab)")
parser.add_argument("--summary-only", action="store_true",
                    help="Skip evaluation; print comparison from existing CSVs")
parser.add_argument("--max-samples", type=int, default=None,
                    help="Evaluate only the first N samples (default: all 1000)")
args = parser.parse_args()

# ── Summary-only mode ─────────────────────────────────────────────────────────
if args.summary_only:
    missing = [p for p in (OUT_CSV_V1, OUT_CSV_V2) if not p.exists()]
    if missing:
        for p in missing:
            log.error(f"Required CSV not found: {p}")
        sys.exit(1)
    df_v1 = pd.read_csv(OUT_CSV_V1)
    df_v2 = pd.read_csv(OUT_CSV_V2)
    n = min(len(df_v1), len(df_v2))
    wf2.print_comparison_summary(df_v1.head(n), df_v2.head(n))
    sys.exit(0)

# ── Full / partial evaluation ─────────────────────────────────────────────────
nlp    = load_spacy()
scorer = NLIScorer(model_id=NLI_MODEL_ID, device=0)

df_v2 = wf2.evaluate_mmfakebench_v2(
    nlp, scorer,
    max_samples=args.max_samples,
    out_csv=OUT_CSV_V2,
    ckpt_csv=CKPT_CSV_V2,
)

print("\n-- v2 output (first 5 rows) --")
cols      = ["idx", "caption", "label", "factcheck_score",
             "status", "n_chunks", "avg_top_sim"]
available = [c for c in cols if c in df_v2.columns]
print(df_v2[available].head().to_string(index=False))

if OUT_CSV_V1.exists():
    df_v1 = pd.read_csv(OUT_CSV_V1)
    n     = min(len(df_v1), len(df_v2))
    wf2.print_comparison_summary(df_v1.head(n), df_v2.head(n))
else:
    log.warning(f"v1 CSV not found at {OUT_CSV_V1} — skipping comparison.")
    from wikipedia_factcheck import print_summary
    print_summary(df_v2.rename(columns={"label": "__label__"}))
