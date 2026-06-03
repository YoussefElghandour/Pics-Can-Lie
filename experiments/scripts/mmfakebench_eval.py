"""
MMFakeBench Evaluation — Ateeqq/ai-vs-human-image-detector
------------------------------------------------------------
Evaluates the AI image detector on:
  - visual_veracity_distortion samples (50 Fakeddit photo-edits + 50 AI-generated)
  - Equal number of real (original) samples for a balanced benchmark

Saves per-image scores to mmfakebench_ai_scores.csv
"""
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402


import os, json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# ── Config ────────────────────────────────────────────────────────────────────
JSON_PATH   = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'MMFakeBench', 'MMFakeBench_val.json')
IMG_ROOT    = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'MMFakeBench', 'MMFakeBench_val')
OUT_CSV     = _os.path.join(str(_cfg.ROOT), 'features', 'mmfakebench_ai_scores.csv')
MODEL_ID    = "Ateeqq/ai-vs-human-image-detector"
BATCH_SIZE  = 16
THRESHOLD   = 0.5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# ── Step 1: Load + inspect JSON ───────────────────────────────────────────────
with open(JSON_PATH) as f:
    data = json.load(f)

print(f"\nTotal annotations: {len(data)}")
print("\nFirst 3 entries:")
for e in data[:3]:
    print(" ", e)

# ── Step 2: Filter samples ────────────────────────────────────────────────────
vvd_samples  = [e for e in data if e["fake_cls"] == "visual_veracity_distortion"]
real_samples = [e for e in data if e["fake_cls"] == "original"]

print(f"\nvisual_veracity_distortion samples : {len(vvd_samples)}")
print(f"original (real) samples            : {len(real_samples)}")

# Balance: take same number of real as fake
n_fake = len(vvd_samples)
np.random.seed(42)
real_indices = np.random.choice(len(real_samples), size=n_fake, replace=False)
real_balanced = [real_samples[i] for i in sorted(real_indices)]

print(f"\nUsing {n_fake} VVD (fake) + {len(real_balanced)} real -> {n_fake + len(real_balanced)} total")

# VVD source breakdown
from collections import Counter
vvd_src = Counter(e["image_source"] for e in vvd_samples)
print(f"VVD breakdown: {dict(vvd_src)}")

# ── Step 3: Load model ────────────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_ID}")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model     = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# Find label indices for "ai" and "hum"
id2label = model.config.id2label
print(f"Model labels: {id2label}")
label2id = {v.lower(): k for k, v in id2label.items()}
ai_idx  = label2id.get("ai",  label2id.get("artificial", 0))
hum_idx = label2id.get("hum", label2id.get("human", 1))
print(f"  ai_idx={ai_idx}  hum_idx={hum_idx}")

# ── Step 4: Inference helper ──────────────────────────────────────────────────
def run_batch(paths):
    """Return (ai_scores, hum_scores) for a list of file paths."""
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            imgs.append(Image.new("RGB", (224, 224), (128, 128, 128)))  # grey placeholder

    inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs[:, ai_idx], probs[:, hum_idx]


# ── Step 5: Process all samples ───────────────────────────────────────────────
all_samples = vvd_samples + real_balanced  # fake first, then real
records = []
skipped = 0

batches = [all_samples[i:i+BATCH_SIZE] for i in range(0, len(all_samples), BATCH_SIZE)]

print(f"\nRunning inference on {len(all_samples)} images (batch={BATCH_SIZE})...")

with tqdm(total=len(all_samples), unit="img") as pbar:
    for batch in batches:
        full_paths = []
        for entry in batch:
            rel = entry["image_path"].lstrip("/\\")
            full_paths.append(os.path.join(IMG_ROOT, rel))

        # Check existence
        missing = [p for p in full_paths if not os.path.exists(p)]
        if missing:
            skipped += len(missing)

        ai_scores, hum_scores = run_batch(full_paths)

        for entry, fpath, ai_s, hum_s in zip(batch, full_paths, ai_scores, hum_scores):
            exists = os.path.exists(fpath)
            pred   = "ai" if ai_s >= THRESHOLD else "hum"
            records.append({
                "image_path":    entry["image_path"],
                "full_path":     fpath,
                "exists":        exists,
                "image_source":  entry["image_source"],
                "gt_answers":    entry["gt_answers"],
                "fake_cls":      entry["fake_cls"],
                "ai_score":      round(float(ai_s),  6),
                "hum_score":     round(float(hum_s), 6),
                "predicted_label": pred,
            })

        pbar.update(len(batch))

print(f"\nSkipped (file not found): {skipped}")

# ── Step 6: Save CSV ──────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df)} rows to {OUT_CSV}")

# ── Step 7: Metrics ───────────────────────────────────────────────────────────
# Ground-truth: fake → 1, real → 0
# Prediction:   "ai"  → 1, "hum" → 0
df_eval = df[df["exists"]].copy()
df_eval["y_true"] = (df_eval["gt_answers"].str.lower() == "fake").astype(int)
df_eval["y_pred"] = (df_eval["predicted_label"] == "ai").astype(int)

y_true = df_eval["y_true"].values
y_pred = df_eval["y_pred"].values

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)
cm   = confusion_matrix(y_true, y_pred)

print("\n" + "="*60)
print("OVERALL RESULTS  (VVD fake + balanced real)")
print("="*60)
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1        : {f1:.4f}  ({f1*100:.2f}%)")
print()
print("Confusion matrix (rows=actual, cols=predicted):")
print("               Pred REAL  Pred AI")
print(f"  Actual REAL     {cm[0][0]:>5}     {cm[0][1]:>5}")
print(f"  Actual FAKE     {cm[1][0]:>5}     {cm[1][1]:>5}")
print()
print("Classification report:")
print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE/AI"]))

# ── Per-group breakdown ───────────────────────────────────────────────────────
def group_metrics(mask, label):
    sub = df_eval[mask]
    if len(sub) == 0:
        print(f"\n  {label}: no samples")
        return
    yt = sub["y_true"].values
    yp = sub["y_pred"].values
    a  = accuracy_score(yt, yp)
    f  = f1_score(yt, yp, zero_division=0)
    p  = precision_score(yt, yp, zero_division=0)
    r  = recall_score(yt, yp, zero_division=0)
    print(f"  {label:<40}  n={len(sub):>3}  acc={a*100:.1f}%  F1={f*100:.1f}%  prec={p:.3f}  rec={r:.3f}")

print("="*60)
print("PER-GROUP BREAKDOWN")
print("="*60)
group_metrics(df_eval["fake_cls"] == "original",                           "Real (original)")
group_metrics(df_eval["image_source"] == "Fakeddit",                       "Fakeddit (photo-edited)")
group_metrics(df_eval["image_source"] == "AI-generated Image",             "AI-generated Image")
group_metrics(df_eval["fake_cls"] == "visual_veracity_distortion",         "All VVD (fake)")
print()

# ── Score distribution summary ────────────────────────────────────────────────
print("="*60)
print("AI SCORE DISTRIBUTION  (mean ± std)")
print("="*60)
for label, mask in [
    ("Real",             df_eval["fake_cls"] == "original"),
    ("Fakeddit edits",   df_eval["image_source"] == "Fakeddit"),
    ("AI-generated",     df_eval["image_source"] == "AI-generated Image"),
]:
    sub = df_eval[mask]["ai_score"]
    print(f"  {label:<25}  mean={sub.mean():.4f}  std={sub.std():.4f}  "
          f"min={sub.min():.4f}  max={sub.max():.4f}")
