"""
Colab path override helper.

Import this module BEFORE calling any evaluation functions to redirect all
Windows-absolute paths (D:\\Pics Can Lie\\...) to Colab's /content/ tree.

Works by monkey-patching module-level constants after import.  Safe because
every path constant is only consumed inside function bodies (or via explicit
default arguments that we override in run_colab.py), not at call sites that
have already captured the old value.

Usage (in run_colab.py or a notebook cell):
    import path_overrides          # patches happen on import
    from path_overrides import OUT_CSV_V2, CKPT_CSV_V2
"""

import sys
from pathlib import Path

COLAB_ROOT = Path("/content")

# Ensure /content is importable
if str(COLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(COLAB_ROOT))

# ── Patch wikipedia_factcheck (v1) ────────────────────────────────────────────
import wikipedia_factcheck as _wf

_wf.PROJECT_ROOT = COLAB_ROOT
_wf.OUT_CSV      = COLAB_ROOT / "mmfakebench_factcheck_scores.csv"
_wf.CKPT_CSV     = COLAB_ROOT / "mmfakebench_factcheck_checkpoint.csv"

# VAL_SEARCH_PATHS is computed at module import time, so patch the list directly.
_wf.VAL_SEARCH_PATHS = [
    COLAB_ROOT / "MMFakeBench" / "val.json",
    COLAB_ROOT / "MMFakeBench" / "val.jsonl",
    COLAB_ROOT / "MMFakeBench" / "data" / "val.json",
    COLAB_ROOT / "dataset" / "MMFakeBench" / "MMFakeBench_val.json",
    COLAB_ROOT / "dataset" / "MMFakeBench" / "val.json",
]

# ── Patch wikipedia_factcheck_v2 ──────────────────────────────────────────────
import wikipedia_factcheck_v2 as _wf2

_wf2.PROJECT_ROOT = COLAB_ROOT
_wf2.OUT_CSV_V2   = COLAB_ROOT / "mmfakebench_factcheck_scores_v2.csv"
_wf2.CKPT_CSV_V2  = COLAB_ROOT / "mmfakebench_factcheck_checkpoint_v2.csv"
_wf2.OUT_CSV_V1   = COLAB_ROOT / "mmfakebench_factcheck_scores.csv"

# ── Convenience exports for notebook / run_colab.py ──────────────────────────
PROJECT_ROOT = COLAB_ROOT
OUT_CSV_V2   = _wf2.OUT_CSV_V2
CKPT_CSV_V2  = _wf2.CKPT_CSV_V2
OUT_CSV_V1   = _wf2.OUT_CSV_V1

print(f"[path_overrides] PROJECT_ROOT => {COLAB_ROOT}")
print(f"[path_overrides] CKPT_CSV_V2  => {CKPT_CSV_V2}")
print(f"[path_overrides] OUT_CSV_V2   => {OUT_CSV_V2}")
print(f"[path_overrides] VAL_JSON     => {_wf.VAL_SEARCH_PATHS[3]}")
