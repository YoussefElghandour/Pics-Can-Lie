"""
BLIP ITM Fine-tuning with Hard Negative Mining
Kaggle Notebook Script
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os, json, time, copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from PIL import Image, UnidentifiedImageError
from transformers import BlipProcessor, BlipForImageTextRetrieval
from sklearn.metrics import f1_score, accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_INPUT   = "/kaggle/input/datasets/youssefelghandour11/bigdataset/kaggle_dataset_full"
IMAGES_ROOT  = f"{BASE_INPUT}/images"
WORKING_DIR  = "/kaggle/working"

TRAIN_LABELS = f"{BASE_INPUT}/merged_balanced/train.json"
VAL_LABELS   = f"{BASE_INPUT}/merged_balanced/val.json"
TRAIN_META   = f"{BASE_INPUT}/metadata/train.json"
VAL_META     = f"{BASE_INPUT}/metadata/val.json"

BEST_CKPT    = f"{WORKING_DIR}/blip_itm_finetuned_best.pt"
HISTORY_JSON = f"{WORKING_DIR}/training_history.json"

MODEL_NAME   = "Salesforce/blip-image-text-matching-base"

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
NUM_WORKERS     = 4
MAX_EPOCHS      = 30
PATIENCE        = 5
LR_HEAD         = 1e-5
LR_ENCODER      = 5e-6
UNFREEZE_LAYERS = 2
HARD_NEG_COPIES = 2          # duplicate hard negatives N times
LOW_CONF_LOW    = 0.4
LOW_CONF_HIGH   = 0.6


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def build_image_path(raw_path: str) -> str:
    """Strip visual_news/ prefix and prepend images root."""
    stripped = raw_path.removeprefix("visual_news/")
    return os.path.join(IMAGES_ROOT, stripped)


def load_split(labels_path: str, meta_path: str, split_name: str):
    """
    Returns a list of dicts: {image_path, caption, label (0/1), image_id}
    Skips entries with missing metadata or non-existent image files.
    """
    with open(labels_path, "r") as f:
        annotations = json.load(f)        # list of {image_id, falsified}

    with open(meta_path, "r") as f:
        metadata = json.load(f)           # dict keyed by str(image_id)

    samples          = []
    missing_meta     = 0
    missing_file     = 0

    for ann in annotations:
        img_id = ann["image_id"]
        key    = str(img_id)

        if key not in metadata:
            missing_meta += 1
            continue

        meta     = metadata[key]
        img_path = build_image_path(meta["image_path"])
        caption  = meta.get("caption", "")

        if not os.path.exists(img_path):
            missing_file += 1
            continue

        samples.append({
            "image_id":   img_id,
            "image_path": img_path,
            "caption":    caption,
            "label":      int(ann["falsified"]),  # 1=fake, 0=real
        })

    print(f"[{split_name}] loaded={len(samples)} | "
          f"missing_meta={missing_meta} | missing_file={missing_file}")
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ══════════════════════════════════════════════════════════════════════════════

class ITMDataset(Dataset):
    def __init__(self, samples, processor, max_text_len=128):
        self.samples     = samples
        self.processor   = processor
        self.max_text_len = max_text_len
        self.corrupt_count = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            image = Image.open(s["image_path"]).convert("RGB")
        except (UnidentifiedImageError, OSError):
            self.corrupt_count += 1
            # return a black placeholder so the batch doesn't crash
            image = Image.new("RGB", (384, 384))

        encoding = self.processor(
            images=image,
            text=s["caption"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
        )
        # squeeze batch dim added by processor
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"]   = torch.tensor(s["label"], dtype=torch.long)
        item["sample_idx"] = torch.tensor(idx, dtype=torch.long)
        return item


def make_loader(samples, processor, shuffle=True, batch_size=BATCH_SIZE):
    ds = ITMDataset(samples, processor)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    ), ds


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL SETUP
# ══════════════════════════════════════════════════════════════════════════════

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_last_n_layers(encoder, n):
    """Unfreeze the last n transformer encoder layers."""
    all_layers = list(encoder.encoder.layer)
    for layer in all_layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True


def build_model_and_optimizer():
    model = BlipForImageTextRetrieval.from_pretrained(MODEL_NAME)

    # Freeze everything first
    freeze_model(model)

    # Unfreeze last 2 layers of vision encoder
    unfreeze_last_n_layers(model.vision_model, UNFREEZE_LAYERS)

    # Unfreeze last 2 layers of text encoder
    unfreeze_last_n_layers(model.text_encoder, UNFREEZE_LAYERS)

    # Always unfreeze ITM head and projection layers
    for name, p in model.named_parameters():
        if any(k in name for k in ["itm_head", "vision_proj", "text_proj"]):
            p.requires_grad = True

    # Parameter groups with different LRs
    encoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and
                      not any(k in n for k in ["itm_head", "vision_proj", "text_proj"])]
    head_params    = [p for n, p in model.named_parameters()
                      if p.requires_grad and
                      any(k in n for k in ["itm_head", "vision_proj", "text_proj"])]

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": LR_ENCODER},
        {"params": head_params,    "lr": LR_HEAD},
    ], weight_decay=1e-4)

    return model, optimizer


# ══════════════════════════════════════════════════════════════════════════════
# 4. HARD NEGATIVE MINING
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def find_hard_samples(model, samples, processor, device):
    """
    Runs inference on all training samples using the full model (DataParallel-
    wrapped or plain) so both GPUs are used during mining.
    Hard = wrong prediction OR low confidence (score in [0.4, 0.6]).
    Returns list of indices into `samples`.
    """
    model.eval()
    ds     = ITMDataset(samples, processor)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    hard_indices = []
    offset = 0

    for batch in loader:
        pixel_values      = batch["pixel_values"].to(device)
        input_ids         = batch["input_ids"].to(device)
        attention_mask    = batch["attention_mask"].to(device)
        labels            = batch["labels"].to(device)

        with autocast():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_itm_head=True,
            )

        logits = outputs.itm_score                      # (B, 2)
        probs  = torch.softmax(logits.float(), dim=-1)  # (B, 2)
        preds  = probs.argmax(dim=-1)                   # (B,)

        # confidence = max prob for the predicted class
        # but we check low-confidence on either class 0 or 1
        fake_prob = probs[:, 1]  # P(fake)

        wrong        = preds != labels
        low_conf     = (fake_prob >= LOW_CONF_LOW) & (fake_prob <= LOW_CONF_HIGH)
        is_hard      = (wrong | low_conf).cpu().numpy()

        for i, hard in enumerate(is_hard):
            if hard:
                hard_indices.append(offset + i)
        offset += len(labels)

    return hard_indices


def build_oversampled_samples(base_samples, hard_indices, copies=HARD_NEG_COPIES):
    """Append `copies` duplicates of hard samples to the base list."""
    hard_samples = [base_samples[i] for i in hard_indices]
    return base_samples + hard_samples * copies


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING / EVAL UTILS
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, optimizer, scaler, device, is_train):
    model.train() if is_train else model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, all_preds, all_labels = 0.0, [], []
    first_batch = True

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            pixel_values   = batch["pixel_values"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with autocast():
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_itm_head=True,
                )
                loss = criterion(outputs.itm_score, labels)

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.itm_score.argmax(dim=-1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            if first_batch and is_train:
                first_batch = False
                for i in range(torch.cuda.device_count()):
                    mem = torch.cuda.memory_allocated(i) / 1e9
                    res = torch.cuda.memory_reserved(i) / 1e9
                    print(f"  GPU {i} after 1st batch: "
                          f"allocated={mem:.2f}GB  reserved={res:.2f}GB")

    n       = len(all_labels)
    avg_loss = total_loss / n
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    return avg_loss, acc, f1


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def check_metadata_structure(meta_path: str):
    """
    Prints the first 5 top-level keys of a metadata JSON so you can confirm
    whether they are image IDs (good) or a wrapper key like 'root' (bad).
    Raises if the structure looks wrong so training never silently produces 0 samples.
    """
    with open(meta_path, "r") as f:
        meta = json.load(f)
    top_keys = list(meta.keys())[:5]
    print(f"[metadata check] {meta_path}")
    print(f"  top-level keys (first 5): {top_keys}")
    if top_keys and not top_keys[0].lstrip("-").isdigit():
        raise ValueError(
            f"metadata top-level key '{top_keys[0]}' is not an image ID. "
            "If the structure is {{\"root\": {{...}}}}, update load_split to use "
            "metadata = json.load(f)[\"root\"] (or the correct wrapper key)."
        )
    print("  structure looks correct — keys are image IDs\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  GPUs: {torch.cuda.device_count()}")

    # ── Validate metadata structure before doing anything expensive ────────
    check_metadata_structure(TRAIN_META)
    check_metadata_structure(VAL_META)

    # ── Load data ──────────────────────────────────────────────────────────
    train_samples = load_split(TRAIN_LABELS, TRAIN_META, "TRAIN")
    val_samples   = load_split(VAL_LABELS,   VAL_META,   "VAL  ")

    processor = BlipProcessor.from_pretrained(MODEL_NAME)

    # ── Model ──────────────────────────────────────────────────────────────
    model, optimizer = build_model_and_optimizer()
    if torch.cuda.device_count() > 1:
        print(f"Wrapping model in DataParallel across "
              f"{torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    scaler = GradScaler()

    # ── Val loader (fixed) ─────────────────────────────────────────────────
    val_loader, _ = make_loader(val_samples, processor, shuffle=False)

    # ── State ──────────────────────────────────────────────────────────────
    history          = []
    best_val_f1      = -1.0
    best_state       = None
    epochs_no_improve = 0
    current_train_samples = train_samples  # may grow each epoch

    print("\n" + "═"*65)
    print("Starting training")
    print("═"*65)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        print(f"\n── Epoch {epoch}/{MAX_EPOCHS} "
              f"| train_set_size={len(current_train_samples)} ──")

        train_loader, train_ds = make_loader(
            current_train_samples, processor, shuffle=True)

        # Train
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, optimizer, scaler, device, is_train=True)

        if train_ds.corrupt_count:
            print(f"  Corrupt images skipped this epoch: "
                  f"{train_ds.corrupt_count}")

        # Validate
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, optimizer, scaler, device, is_train=False)

        elapsed = time.time() - t0
        print(f"  train_loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_F1={val_f1:.4f}  "
              f"({elapsed:.0f}s)")

        # ── Hard negative mining (epoch 1 is warmup, mining starts epoch 2) ──
        num_hard      = 0
        next_train_sz = len(train_samples)

        if epoch >= 2:
            hard_indices = find_hard_samples(
                model, train_samples, processor, device)
            num_hard = len(hard_indices)

            if hard_indices:
                current_train_samples = build_oversampled_samples(
                    train_samples, hard_indices)
            else:
                current_train_samples = train_samples

            next_train_sz = len(current_train_samples)
            print(f"  hard_samples={num_hard} | "
                  f"next_epoch_train_size={next_train_sz}")

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            inner = model.module if hasattr(model, "module") else model
            best_state = copy.deepcopy(inner.state_dict())
            torch.save(best_state, BEST_CKPT)
            print(f"  ✓ New best val F1={best_val_f1:.4f} — checkpoint saved")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{PATIENCE})")

        # ── Record history ─────────────────────────────────────────────────
        history.append({
            "epoch":            epoch,
            "train_loss":       round(train_loss, 6),
            "train_acc":        round(train_acc, 6),
            "train_f1":         round(train_f1, 6),
            "val_loss":         round(val_loss, 6),
            "val_acc":          round(val_acc, 6),
            "val_f1":           round(val_f1, 6),
            "hard_samples":     num_hard,
            "next_train_size":  next_train_sz,
            "elapsed_sec":      round(elapsed, 1),
        })

        with open(HISTORY_JSON, "w") as f:
            json.dump(history, f, indent=2)

        # ── Early stopping ─────────────────────────────────────────────────
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "═"*80)
    print(f"{'Epoch':>6} {'TrainLoss':>10} {'ValAcc':>8} {'ValF1':>8} "
          f"{'Hard':>8} {'NextSz':>8} {'Time':>7}")
    print("─"*80)
    for r in history:
        print(f"{r['epoch']:>6} {r['train_loss']:>10.4f} "
              f"{r['val_acc']:>8.4f} {r['val_f1']:>8.4f} "
              f"{r['hard_samples']:>8} {r['next_train_size']:>8} "
              f"{r['elapsed_sec']:>6.0f}s")
    print("═"*80)
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Best checkpoint: {BEST_CKPT}")
    print(f"Training history: {HISTORY_JSON}")


if __name__ == "__main__":
    main()
