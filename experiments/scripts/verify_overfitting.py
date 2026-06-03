"""
Overfitting verification for fine-tuned ai_detector model.

Tests:
  A) Training history — printed from known epoch log
  B) 200 new real NewsClipPings images (seed=99, no overlap with training seed=42)
  C) 100 real samples from NewsClipPings val.json (held-out split)

For B and C: run BOTH original and fine-tuned model to get before/after FP rates.
"""
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402


import random, json, warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

SEED_TRAIN = 42
SEED_NEW   = 99
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD  = 0.5
BATCH_SIZE = 16

MODEL_ORIG = "Ateeqq/ai-vs-human-image-detector"
MODEL_FT   = _os.path.join(str(_cfg.ROOT), 'models', 'ai_detector_finetuned')
NC_ROOT    = Path(_os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'origin', 'origin'))
MERGED_VAL = Path(_os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'merged_balanced', 'val.json'))
META_VAL   = Path(_os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'metadata', 'val.json'))
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

print(f"Device: {DEVICE}")

# ── Section 1: Training history ────────────────────────────────────────────────
print("\n" + "="*66)
print("  SECTION 1 — TRAINING HISTORY (from training run)")
print("="*66)
print("""
  Phase 1 — Classifier only (LR=1e-4, trainable params=1,538)
  +---------+-------------+----------+---------+---------+
  | Epoch   | Train Loss  | Train Acc| Val Loss | Val Acc |
  +---------+-------------+----------+---------+---------+
  |   1/3   |   2.6693    |  44.62%  |  1.2158 |  47.00% |
  |   2/3   |   1.0225    |  49.62%  |  0.9698 |  54.00% |
  |   3/3   |   0.9549    |  50.88%  |  0.9388 |  54.50% |*best
  +---------+-------------+----------+---------+---------+

  Phase 2 — Classifier + last 2 encoder layers (LR=1e-5, trainable=14,177,282)
  +---------+-------------+----------+---------+---------+
  | Epoch   | Train Loss  | Train Acc| Val Loss | Val Acc |
  +---------+-------------+----------+---------+---------+
  |   1/3   |   0.7426    |  60.62%  |  0.6001 |  69.00% |
  |   2/3   |   0.4864    |  76.12%  |  0.4410 |  79.50% |
  |   3/3   |   0.3508    |  85.62%  |  0.3679 |  84.50% |*best
  +---------+-------------+----------+---------+---------+

  Observations:
  - Phase 1: Train/val both low and close — classifier alone has little capacity.
  - Phase 2: Val accuracy tracks train closely each epoch (no divergence):
      Epoch 1: train=60.6%  val=69.0%  (val ABOVE train — model generalising)
      Epoch 2: train=76.1%  val=79.5%  (val still above train)
      Epoch 3: train=85.6%  val=84.5%  (1.1pp gap — negligible)
  - Val loss decreases monotonically: 0.60 -> 0.44 -> 0.37 (no uptick = no overfit)
  - Val accuracy never peaked then dropped (early stopping never triggered).
  - Training history alone does NOT indicate overfitting.
""")

# ── Collect all real images from NewsClipPings origin ─────────────────────────
print("="*66)
print("  Collecting all real images from NewsClipPings origin...")
print("="*66)
nc_sources = ["bbc", "guardian", "usa_today", "washington_post"]
all_real = []
for src in nc_sources:
    imgs = [str(p) for p in (NC_ROOT / src).rglob("*")
            if p.suffix.lower() in IMG_EXTS]
    print(f"  {src}: {len(imgs):,}")
    all_real.extend(imgs)
print(f"  Total: {len(all_real):,}")

# Reproduce training sample (seed=42) so we can exclude it
random.seed(SEED_TRAIN)
all_real_copy = all_real.copy()
random.shuffle(all_real_copy)
train_set = set(all_real_copy[:500])
print(f"  Training set (seed={SEED_TRAIN}): {len(train_set)} images")

# Sample 200 new images (seed=99), excluding training set
random.seed(SEED_NEW)
candidates = [p for p in all_real if p not in train_set]
random.shuffle(candidates)
new_real_paths = candidates[:200]
overlap = len([p for p in new_real_paths if p in train_set])
print(f"  New sample  (seed={SEED_NEW}): {len(new_real_paths)} images  "
      f"(overlap with training: {overlap})")

# ── Load NewsClipPings val set — 100 real samples ─────────────────────────────
print("\n" + "="*66)
print("  Loading NewsClipPings val.json real samples...")
print("="*66)
with open(MERGED_VAL) as f:
    val_data = json.load(f)
with open(META_VAL) as f:
    val_meta = json.load(f)

random.seed(SEED_NEW)
real_anns = [e for e in val_data["annotations"] if not e["falsified"]]
random.shuffle(real_anns)

val_paths_resolved = []
for ann in real_anns:
    img_id = str(ann["image_id"])
    meta   = val_meta.get(img_id, {})
    raw    = meta.get("image_path", "")
    # visual_news/origin/bbc/... → dataset/origin/origin/bbc/...
    rel = raw.replace("visual_news/origin/", "")
    full = NC_ROOT / rel
    if full.exists():
        val_paths_resolved.append(str(full))
    if len(val_paths_resolved) == 100:
        break

print(f"  Resolved {len(val_paths_resolved)} / 100 requested real val images")

# ── Load models ────────────────────────────────────────────────────────────────
def load_model_and_processor(model_id, label):
    print(f"\nLoading {label} ...")
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    model     = AutoModelForImageClassification.from_pretrained(model_id).to(DEVICE)
    model.eval()
    return model, processor

model_orig, proc_orig = load_model_and_processor(MODEL_ORIG, "original model")
model_ft,   proc_ft   = load_model_and_processor(MODEL_FT,   "fine-tuned model")

# ── Inference helper ───────────────────────────────────────────────────────────
def infer_fp_rate(model, processor, paths, label):
    """Run inference on a list of REAL image paths; return FP rate (pred=ai on real)."""
    ai_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), BATCH_SIZE),
                      desc=f"  {label}", unit="batch", leave=False):
            batch = paths[i:i+BATCH_SIZE]
            imgs  = []
            for p in batch:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except (OSError, UnidentifiedImageError):
                    imgs.append(Image.new("RGB", (224, 224), (128, 128, 128)))
            inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            # id2label: 0=ai, 1=hum
            ai_scores = probs[:, 0]
            ai_preds.extend((ai_scores >= THRESHOLD).tolist())
    n_fp  = sum(ai_preds)
    fp_rt = n_fp / len(ai_preds) if ai_preds else float("nan")
    return fp_rt, n_fp, len(ai_preds)

# ── Section 2: New NewsClipPings sample (seed=99) ─────────────────────────────
print("\n" + "="*66)
print("  SECTION 2 — New real images, seed=99 (200 images, not in training)")
print("="*66)
fp_new_orig, n_fp_new_orig, n_new = infer_fp_rate(
    model_orig, proc_orig, new_real_paths, "original model")
fp_new_ft,   n_fp_new_ft,   _     = infer_fp_rate(
    model_ft,   proc_ft,   new_real_paths, "fine-tuned model")
print(f"  Original  : FP rate = {fp_new_orig*100:.1f}%  ({n_fp_new_orig}/{n_new} misclassified as AI)")
print(f"  Fine-tuned: FP rate = {fp_new_ft*100:.1f}%  ({n_fp_new_ft}/{n_new} misclassified as AI)")

# ── Section 3: NewsClipPings val set ──────────────────────────────────────────
print("\n" + "="*66)
print("  SECTION 3 — NewsClipPings val.json real samples (100 images)")
print("="*66)
fp_val_orig, n_fp_val_orig, n_val = infer_fp_rate(
    model_orig, proc_orig, val_paths_resolved, "original model")
fp_val_ft,   n_fp_val_ft,   _     = infer_fp_rate(
    model_ft,   proc_ft,   val_paths_resolved, "fine-tuned model")
print(f"  Original  : FP rate = {fp_val_orig*100:.1f}%  ({n_fp_val_orig}/{n_val} misclassified as AI)")
print(f"  Fine-tuned: FP rate = {fp_val_ft*100:.1f}%  ({n_fp_val_ft}/{n_val} misclassified as AI)")

# ── Summary table ──────────────────────────────────────────────────────────────
# Known baselines from the held-out MMFakeBench eval
FP_MMFAKE_BEFORE = 0.479   # 34/71 VisualNews real
FP_MMFAKE_AFTER  = 0.296   # 21/71 VisualNews real

print("\n" + "="*70)
print("  OVERFITTING VERDICT — FALSE POSITIVE RATE COMPARISON")
print("="*70)
print(f"  {'Test Set':<45} {'FP Before':>10} {'FP After':>10}")
print(f"  {'-'*65}")
print(f"  {'MMFakeBench val VisualNews (held-out, original eval)':<45} "
      f"{FP_MMFAKE_BEFORE*100:>9.1f}% {FP_MMFAKE_AFTER*100:>9.1f}%")
print(f"  {'New NewsClipPings sample (seed=99, 200 imgs)':<45} "
      f"{fp_new_orig*100:>9.1f}% {fp_new_ft*100:>9.1f}%")
print(f"  {'NewsClipPings val.json real (100 imgs)':<45} "
      f"{fp_val_orig*100:>9.1f}% {fp_val_ft*100:>9.1f}%")
print(f"  {'-'*65}")

# Verdict
after_rates = [FP_MMFAKE_AFTER, fp_new_ft, fp_val_ft]
spread = max(after_rates) - min(after_rates)
avg_after = sum(after_rates) / len(after_rates)
print(f"\n  Fine-tuned FP rates: {[f'{r*100:.1f}%' for r in after_rates]}")
print(f"  Spread (max-min)   : {spread*100:.1f}pp")
print(f"  Average after      : {avg_after*100:.1f}%")
print()
if spread < 0.12:
    verdict = "GENERALISATION CONFIRMED — FP rates consistent across all test sets."
elif spread < 0.20:
    verdict = "MILD VARIANCE — Some spread across test sets; likely not memorisation."
else:
    verdict = "WARNING — Large spread suggests possible overfitting. Investigate further."
print(f"  Verdict: {verdict}")
print("="*70)
