"""
Fine-tune Ateeqq/ai-vs-human-image-detector (SiglipForImageClassification)
on domain-specific data to reduce false positives on real news photos.

Two-phase training
  Phase 1 — classifier only   : 3 epochs, LR 1e-4
  Phase 2 — classifier + last 2 encoder layers : 3 epochs, LR 1e-5

Training data  (no overlap with eval set)
  Real (label=hum): 500 randomly sampled from NewsClipPings origin folders
  Fake (label=ai) : 250 from antifact_image_generation_test_500 (AI-generated)
                  + 250 from Fakeddit_photo_edit_test_500 (PS-edited)

Eval set  (completely held-out, never seen during training)
  mmfakebench_ai_scores.csv  →  200 images from MMFakeBench val set
"""
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402


import os, random, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import copy

warnings.filterwarnings("ignore", category=UserWarning)

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID       = "Ateeqq/ai-vs-human-image-detector"
PROJECT_ROOT   = Path(str(_cfg.ROOT))
NC_ORIGIN_ROOT = PROJECT_ROOT / "dataset" / "origin" / "origin"
TEST_FAKE_ROOT = PROJECT_ROOT / "dataset" / "MMFakeBench" / "MMFakeBench_test" / "fake"
EVAL_CSV       = PROJECT_ROOT / "mmfakebench_ai_scores.csv"
EVAL_IMG_ROOT  = PROJECT_ROOT / "dataset" / "MMFakeBench" / "MMFakeBench_val"
SAVE_DIR       = PROJECT_ROOT / "ai_detector_finetuned"
OUT_CSV        = PROJECT_ROOT / "mmfakebench_ai_scores_finetuned.csv"

N_REAL         = 500
N_AI_FAKE      = 250
N_PS_FAKE      = 250
BATCH_SIZE     = 16
VAL_FRAC       = 0.20
THRESHOLD      = 0.5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device : {DEVICE}")
print(f"Seed   : {SEED}")

# ── 1. Collect image paths ─────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def collect_images(folder: Path) -> list:
    return [str(p) for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]

print("\nCollecting real images from NewsClipPings...")
nc_sources = ["bbc", "guardian", "usa_today", "washington_post"]
all_real = []
for src in nc_sources:
    imgs = collect_images(NC_ORIGIN_ROOT / src)
    print(f"  {src}: {len(imgs):,}")
    all_real.extend(imgs)
print(f"  Total available: {len(all_real):,}")

random.shuffle(all_real)
real_paths = all_real[:N_REAL]
print(f"  Sampled: {len(real_paths)}")

print("\nCollecting fake images from MMFakeBench TEST set...")
ai_fake_dir = TEST_FAKE_ROOT / "antifact_image_generation_test_500"
ps_fake_dir = TEST_FAKE_ROOT / "Fakeddit_photo_edit_test_500"

all_ai = collect_images(ai_fake_dir)
all_ps = collect_images(ps_fake_dir)
random.shuffle(all_ai)
random.shuffle(all_ps)
ai_paths = all_ai[:N_AI_FAKE]
ps_paths = all_ps[:N_PS_FAKE]
print(f"  AI-generated (antifact): {len(all_ai)} available, using {len(ai_paths)}")
print(f"  PS-edited  (Fakeddit) : {len(all_ps)} available, using {len(ps_paths)}")

fake_paths = ai_paths + ps_paths
print(f"  Total fake: {len(fake_paths)}")

# ── 2. Build balanced dataset with 80/20 stratified split ──────────────────────
def split_group(paths, val_frac, shuffle=True):
    if shuffle:
        paths = paths.copy()
        random.shuffle(paths)
    n_val = max(1, int(len(paths) * val_frac))
    return paths[n_val:], paths[:n_val]  # train, val

real_train, real_val     = split_group(real_paths, VAL_FRAC)
ai_train,   ai_val       = split_group(ai_paths,   VAL_FRAC)
ps_train,   ps_val       = split_group(ps_paths,   VAL_FRAC)

train_paths  = real_train + ai_train + ps_train
train_labels = [1]*len(real_train) + [0]*len(ai_train) + [0]*len(ps_train)
val_paths    = real_val + ai_val + ps_val
val_labels   = [1]*len(real_val)   + [0]*len(ai_val)   + [0]*len(ps_val)

# Shuffle train set
combined = list(zip(train_paths, train_labels))
random.shuffle(combined)
train_paths, train_labels = zip(*combined)
train_paths, train_labels = list(train_paths), list(train_labels)

print(f"\nDataset split:")
print(f"  Train: {len(train_paths)}  "
      f"(real={sum(l==1 for l in train_labels)}, fake={sum(l==0 for l in train_labels)})")
print(f"  Val  : {len(val_paths)}  "
      f"(real={sum(l==1 for l in val_labels)}, fake={sum(l==0 for l in val_labels)})")

# ── 3. Dataset ─────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, paths, labels, processor):
        self.paths     = paths
        self.labels    = labels
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except (OSError, UnidentifiedImageError):
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        encoded = self.processor(images=img, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels":       torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── 4. Load model & processor ──────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_ID}")
processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
model     = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(DEVICE)

# id2label: {0: 'ai', 1: 'hum'}  →  training label 0=fake/ai, 1=real/hum
print(f"id2label: {model.config.id2label}")
print(f"Encoder layers: {len(model.vision_model.encoder.layers)}")

train_ds = ImageDataset(train_paths, train_labels, processor)
val_ds   = ImageDataset(val_paths,   val_labels,   processor)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=0, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=True)

# ── 5. Training helpers ────────────────────────────────────────────────────────
def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            pv = batch["pixel_values"].to(DEVICE)
            lb = batch["labels"].to(DEVICE)
            logits = model(pixel_values=pv).logits
            loss   = criterion(logits, lb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(lb)
            preds_all.extend(logits.argmax(dim=-1).cpu().tolist())
            labels_all.extend(lb.cpu().tolist())
    avg_loss = total_loss / len(labels_all)
    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, zero_division=0,
                   labels=[0, 1], average="macro")
    return avg_loss, acc, f1

def train_phase(model, phase_name, n_epochs, lr, patience=2):
    print(f"\n{'='*60}")
    print(f"  {phase_name}  |  trainable params: {count_trainable(model):,}  |  LR={lr}")
    print(f"{'='*60}")
    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion  = nn.CrossEntropyLoss()
    best_acc   = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_dl, optimizer, criterion, train=True)
        vl_loss, vl_acc, vl_f1 = run_epoch(model, val_dl,   optimizer, criterion, train=False)
        print(f"  Epoch {epoch}/{n_epochs}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}%  |  "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc*100:.2f}% val_F1={vl_f1*100:.2f}%")

        if vl_acc > best_acc:
            best_acc   = vl_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            print(f"    ** New best val_acc={best_acc*100:.2f}% — checkpoint saved")
        else:
            no_improve += 1
            print(f"    No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"    Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    print(f"  Best val_acc for {phase_name}: {best_acc*100:.2f}%")
    return best_acc

# ── Phase 1: Classifier only ───────────────────────────────────────────────────
freeze_all(model)
unfreeze(model.classifier)
best_p1 = train_phase(model, "Phase 1 — Classifier only", n_epochs=3, lr=1e-4, patience=2)

# ── Phase 2: Classifier + last 2 encoder layers ───────────────────────────────
for layer in model.vision_model.encoder.layers[-2:]:
    unfreeze(layer)
best_p2 = train_phase(model, "Phase 2 — Classifier + last 2 encoder layers",
                      n_epochs=3, lr=1e-5, patience=2)

# ── 6. Save fine-tuned model ───────────────────────────────────────────────────
print(f"\nSaving fine-tuned model to {SAVE_DIR} ...")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(SAVE_DIR))
processor.save_pretrained(str(SAVE_DIR))
print("Model saved.")

# ── 7. Evaluate on held-out mmfakebench_ai_scores.csv ─────────────────────────
print(f"\n{'='*60}")
print("  HELD-OUT EVALUATION  (mmfakebench_ai_scores.csv)")
print(f"{'='*60}")

df_eval = pd.read_csv(EVAL_CSV)
print(f"Eval samples: {len(df_eval)}")

model.eval()
new_ai_scores, new_hum_scores, new_preds = [], [], []

# Batch inference
paths_eval = [str(EVAL_IMG_ROOT / row["image_path"].lstrip("/\\"))
              for _, row in df_eval.iterrows()]

with torch.no_grad():
    for i in tqdm(range(0, len(paths_eval), BATCH_SIZE), desc="Eval inference", unit="batch"):
        batch_paths = paths_eval[i:i+BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224), (128, 128, 128)))
        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        # id2label: 0=ai, 1=hum
        new_ai_scores.extend(probs[:, 0].tolist())
        new_hum_scores.extend(probs[:, 1].tolist())
        new_preds.extend(["ai" if p >= THRESHOLD else "hum" for p in probs[:, 0]])

df_eval["ai_score_ft"]       = [round(s, 6) for s in new_ai_scores]
df_eval["hum_score_ft"]      = [round(s, 6) for s in new_hum_scores]
df_eval["predicted_label_ft"] = new_preds
df_eval.to_csv(OUT_CSV, index=False)
print(f"New scores saved to {OUT_CSV}")

# ── Metric helpers ─────────────────────────────────────────────────────────────
def metrics_report(mask, label, df):
    sub = df[mask].copy()
    if len(sub) == 0:
        print(f"  {label}: no samples"); return
    y_true = (sub["gt_answers"].str.lower() == "fake").astype(int).values
    y_pred = (sub["predicted_label_ft"] == "ai").astype(int).values
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    pr  = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return acc, f1, pr, rec, len(sub)

y_true_all = (df_eval["gt_answers"].str.lower() == "fake").astype(int).values
y_pred_all = (df_eval["predicted_label_ft"] == "ai").astype(int).values
ov_acc = accuracy_score(y_true_all, y_pred_all)
ov_f1  = f1_score(y_true_all, y_pred_all, zero_division=0)

# False positive rate on VisualNews real images
vn_real = df_eval[(df_eval["image_source"] == "VisualNews") &
                  (df_eval["gt_answers"].str.lower() == "true")]
vn_fp_rate = (vn_real["predicted_label_ft"] == "ai").mean()

# PS-edited detection rate  (Fakeddit, gt=Fake)
ps_fake = df_eval[(df_eval["image_source"] == "Fakeddit") &
                  (df_eval["gt_answers"].str.lower() == "fake")]
ps_detect = (ps_fake["predicted_label_ft"] == "ai").mean() if len(ps_fake) > 0 else float("nan")

# AI-generated detection rate (fever_AI_val_100, gt=Fake)
ai_gen = df_eval[(df_eval["image_source"] == "AI-generated Image") &
                 (df_eval["gt_answers"].str.lower() == "fake")]
ai_detect = (ai_gen["predicted_label_ft"] == "ai").mean() if len(ai_gen) > 0 else float("nan")

# ── Final comparison table ─────────────────────────────────────────────────────
print(f"\n{'='*66}")
print(f"  BEFORE vs AFTER FINE-TUNING")
print(f"{'='*66}")
print(f"  {'Metric':<42} {'Before':>9}  {'After':>9}  {'Delta':>8}")
print(f"  {'-'*62}")

def delta(after, before, pct=True):
    d = (after - before)
    sign = "+" if d >= 0 else ""
    return f"{sign}{d*100:.1f}pp" if pct else f"{sign}{d:.3f}"

rows = [
    ("Overall accuracy",            0.5600, ov_acc),
    ("Overall F1",                   0.5217, ov_f1),
    ("FP rate on VisualNews real",   0.4790, vn_fp_rate),
    ("PS-edited detection rate",     0.2400, ps_detect),
    ("AI-generated detection rate",  0.7200, ai_detect),
]
for label, before, after in rows:
    d = delta(after, before)
    print(f"  {label:<42} {before*100:>8.1f}%  {after*100:>8.1f}%  {d:>8}")

print(f"{'='*66}")
print(f"\n  VN false-positive sample counts:")
print(f"    Total VisualNews real : {len(vn_real)}")
print(f"    Predicted AI (FP)     : {(vn_real['predicted_label_ft']=='ai').sum()}")
print(f"  PS-edited detection:")
print(f"    Total PS-edited fake  : {len(ps_fake)}")
print(f"    Detected as AI        : {(ps_fake['predicted_label_ft']=='ai').sum()}")
print(f"  AI-generated detection:")
print(f"    Total AI-gen fake     : {len(ai_gen)}")
print(f"    Detected as AI        : {(ai_gen['predicted_label_ft']=='ai').sum()}")
