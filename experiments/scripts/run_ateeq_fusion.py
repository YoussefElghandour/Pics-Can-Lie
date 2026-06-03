"""
Run fine-tuned Ateeqq AI detector on the 300-sample NewsClipPings fusion set,
add scores to fusion_features_300_3feat_finetuned.csv, then re-run ablation
comparing SightEngine vs Ateeqq as the AI detection module.
"""
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402


import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(str(_cfg.ROOT))
FEATURES_CSV  = PROJECT_ROOT / "fusion_features_300_3feat_finetuned.csv"
ATEEQ_MODEL   = PROJECT_ROOT / "ai_detector_finetuned"
ORIGIN_BASE   = PROJECT_ROOT / "dataset" / "origin"
OUT_CSV       = PROJECT_ROOT / "fusion_features_300_ateeq.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Load all metadata splits ───────────────────────────────────────────────────
meta_all = {}
for split in ["train", "val", "test"]:
    meta_path = PROJECT_ROOT / "dataset" / "data" / "NewsClipPings" / "metadata" / f"{split}.json"
    with open(meta_path) as f:
        meta_all.update(json.load(f))
print(f"Metadata loaded: {len(meta_all):,} entries")

def resolve_image_path(image_id: int) -> Path | None:
    entry = meta_all.get(str(image_id))
    if entry is None:
        return None
    rel = entry["image_path"].replace("visual_news/", "", 1)
    return ORIGIN_BASE / rel

# ── Load Ateeqq fine-tuned model ───────────────────────────────────────────────
print("Loading fine-tuned Ateeqq model...")
processor = AutoImageProcessor.from_pretrained(str(ATEEQ_MODEL))
model = AutoModelForImageClassification.from_pretrained(str(ATEEQ_MODEL)).to(DEVICE).eval()

# Verify label mapping: id2label
print("Label map:", model.config.id2label)

def get_ateeq_score(img_path: Path) -> float:
    """Return probability that the image is AI-generated (label='ai')."""
    try:
        img = Image.open(img_path).convert("RGB")
    except (UnidentifiedImageError, Exception):
        return -1.0
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    # Find the 'ai' label index
    ai_idx = next(
        (k for k, v in model.config.id2label.items() if "ai" in v.lower()),
        1  # fallback
    )
    return probs[ai_idx].item()

# ── Run inference on 300 samples ──────────────────────────────────────────────
df = pd.read_csv(FEATURES_CSV)
print(f"Loaded {len(df)} samples")

scores = []
missing = 0
for _, row in tqdm(df.iterrows(), total=len(df), desc="Ateeqq inference"):
    img_path = resolve_image_path(row["image_id"])
    if img_path is None or not img_path.exists():
        scores.append(-1.0)
        missing += 1
    else:
        scores.append(get_ateeq_score(img_path))

df["ateeq_score"] = scores
print(f"\nMissing images: {missing}/{len(df)}")
print(f"Score stats:\n{df['ateeq_score'].describe()}")

df.to_csv(OUT_CSV, index=False)
print(f"Saved to {OUT_CSV}")

# ── Ablation: SightEngine vs Ateeqq as AI detection module ───────────────────
print("\n" + "="*75)
print("ABLATION: SightEngine vs Ateeqq as AI detection module")
print("="*75)

# Drop rows with missing Ateeqq scores
df_valid = df[df["ateeq_score"] >= 0].copy()
print(f"Valid rows for Ateeqq ablation: {len(df_valid)}")

y = df_valid["label"].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "f1", "precision", "recall"]

COMBINATIONS = [
    ("BLIP only",                         df_valid[["itm_score"]].values),
    ("DeBERTa only",                      df_valid[["entailment_score"]].values),
    ("BLIP + SightEngine",                df_valid[["itm_score", "sightengine_score"]].values),
    ("BLIP + Ateeqq",                     df_valid[["itm_score", "ateeq_score"]].values),
    ("BLIP + DeBERTa",                    df_valid[["itm_score", "entailment_score"]].values),
    ("BLIP + DeBERTa + SightEngine",      df_valid[["itm_score", "entailment_score", "sightengine_score"]].values),
    ("BLIP + DeBERTa + Ateeqq",           df_valid[["itm_score", "entailment_score", "ateeq_score"]].values),
]

print(f"\n{'Combination':<35} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
print("-"*75)

records = []
for name, X in COMBINATIONS:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
    ])
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    acc  = res["test_accuracy"].mean()
    f1   = res["test_f1"].mean()
    prec = res["test_precision"].mean()
    rec  = res["test_recall"].mean()
    records.append(dict(combination=name, accuracy=round(acc,4), f1=round(f1,4),
                        precision=round(prec,4), recall=round(rec,4)))
    print(f"{name:<35} {acc*100:>9.2f}% {f1*100:>9.2f}% {prec*100:>9.2f}% {rec*100:>9.2f}%")

# Save results
out = {
    "description": "Ablation comparing SightEngine vs Ateeqq as AI module, n=300, 5-fold CV",
    "n_samples": len(df_valid),
    "results": records,
}
out_json = PROJECT_ROOT / "ablation_ateeq_vs_sightengine.json"
with open(out_json, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_json}")
