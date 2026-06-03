"""
Central path config for Pics Can Lie.

One environment variable, PCL_ROOT, locates the repo; every dataset / model /
feature / cache / results path is derived relative to it, so the project runs on
any machine (the 4060, a 4090, a clean GitHub clone) by setting PCL_ROOT.

MMFakeBench lives on a separate drive in the original setup; override with
PCL_MMFB_ROOT. Datasets default under <root>/datasets; override PCL_DATA_ROOT.
"""
from __future__ import annotations

import os
from pathlib import Path

# Single source of truth. Default keeps the original D:\ location working.
ROOT = Path(os.environ.get("PCL_ROOT", r"D:\Pics Can Lie"))

# Top-level areas (post-reorg layout).
SRC        = ROOT / "src"
SCRIPTS    = ROOT / "scripts"
EXPERIMENTS = ROOT / "experiments"
MODELS     = ROOT / "models"
FEATURES   = ROOT / "features"
RESULTS    = ROOT / "results"
CACHE      = ROOT / "cache"
DOCS       = ROOT / "docs"
ARCHIVE    = ROOT / "archive"

# Datasets (large; gitignored). MMFakeBench may live on another drive.
DATA_ROOT  = Path(os.environ.get("PCL_DATA_ROOT", ROOT / "datasets"))
MMFB_ROOT  = Path(os.environ.get("PCL_MMFB_ROOT", r"E:\Pics Can Lie\dataset\MMFakeBench"))

# NewsCLIPpings dataset paths (under DATA_ROOT/dataset).
NEWSCLIP        = DATA_ROOT / "dataset" / "data" / "NewsClipPings"
TEST_ANN        = NEWSCLIP / "merged_balanced" / "test.json"
TEST_META       = NEWSCLIP / "metadata" / "test.json"
VAL_ANN         = NEWSCLIP / "merged_balanced" / "val.json"
VAL_META        = NEWSCLIP / "metadata" / "val.json"
IMAGES_ROOT     = DATA_ROOT / "dataset" / "origin" / "origin"
ARTICLE_BASE    = DATA_ROOT / "dataset" / "origin"

# Model checkpoints / artifacts.
CLIP_CKPT   = MODELS / "clip_finetuned_v2" / "clip_classifier.pt"
CLIP_VAL_FEATURES = MODELS / "clip_finetuned_v2" / "val_features"
AITR_CKPT   = MODELS / "fusion_aitr" / "aitr_weights.pt"
SCALER_PATH = MODELS / "fusion_aitr" / "scalar_scaler.joblib"
THRESH_PATH = MODELS / "fusion_aitr" / "frozen_threshold.json"
GBM_PATH    = MODELS / "mmfakebench_gbm.joblib"

# MMFakeBench training feature dir (large; under experiments).
MMFB_TRAIN  = EXPERIMENTS / "mmfakebench_training"

# Feature score files.
DEBERTA_V2_CSV = FEATURES / "deberta_val_scores_v2.csv"
DEBERTA_V3_CSV = FEATURES / "deberta_val_scores_v3.csv"
WIKI_NLI_CSV   = FEATURES / "wiki_nli_scores.csv"
EVIDENCE_CSV   = FEATURES / "evidence_clip_scores.csv"
VAL_SAMPLE_IDS = FEATURES / "val_sample_ids.csv"


def p(*parts) -> Path:
    """Convenience: build a path under ROOT."""
    return ROOT.joinpath(*parts)
