"""
BLIP-2 fine-tuning script — Pics Can Lie
Model  : Salesforce/blip2-opt-2.7b
Task   : Text generation — predict 'real' or 'out-of-context'
Dataset: D:/Pics Can Lie/merged_balanced/train.json

Usage:
    python train_blip2_local.py
"""
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402


import json
import os
import time

import torch
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "Salesforce/blip2-opt-2.7b"
ANNOTATIONS     = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'merged_balanced', 'train.json')
METADATA        = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'metadata', 'train.json')
IMAGES_ROOT     = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'origin')
OUTPUT_DIR      = _os.path.join(str(_cfg.ROOT), 'models', 'blip2_finetuned')
MAX_SAMPLES     = 5000
BATCH_SIZE      = 2
GRAD_ACCUM      = 8          # effective batch = 16
EPOCHS          = 2
LR_QFORMER      = 1e-5
LR_VISION       = 1e-6
VISION_UNFREEZE = 2          # last N layers of vision encoder
LOG_EVERY       = 50         # steps
MAX_TARGET_LEN  = 8          # 'real' or 'out-of-context' + EOS
MAX_INPUT_LEN   = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PicsCanLieDataset(Dataset):
    def __init__(self, annotations_path: str, metadata_path: str,
                 images_root: str, max_samples: int):
        with open(annotations_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)["annotations"]
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)  # keyed by image_id (string)

        joined = []
        for ann in annotations:
            image_id = str(ann["image_id"])
            if image_id not in metadata:
                continue
            meta    = metadata[image_id]
            caption = meta.get("caption") or meta.get("title") or ""
            rel     = meta["image_path"].replace("visual_news/", "", 1)
            joined.append({
                "image_path": os.path.join(images_root, rel),
                "caption":    caption,
                "label_text": "out-of-context" if ann["falsified"] else "real",
            })

        self.items = joined[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        prompt = (
            f"Does this image match the caption: '{item['caption'][:100]}'? "
            "Answer:"
        )
        return image, prompt, item["label_text"]


def collate_fn(batch, processor):
    images, prompts, labels = zip(*batch)

    # Tokenise prompts (input)
    inputs = processor(
        images=list(images),
        text=list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LEN,
    )

    # Tokenise labels (target); pad to fixed length so we can mask
    label_enc = processor.tokenizer(
        list(labels),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_TARGET_LEN,
    )
    label_ids = label_enc["input_ids"].clone()
    # Replace padding token id with -100 so loss ignores padding
    label_ids[label_ids == processor.tokenizer.pad_token_id] = -100

    inputs["labels"] = label_ids
    return inputs


# ── Freeze / unfreeze helpers ─────────────────────────────────────────────────

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_qformer(model):
    for p in model.qformer.parameters():
        p.requires_grad = True
    for p in model.language_projection.parameters():
        p.requires_grad = True


def unfreeze_vision_last_n(model, n: int):
    encoder_layers = model.vision_model.encoder.layers
    for layer in encoder_layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True


def build_optimizer(model):
    qformer_params  = list(model.qformer.parameters()) + \
                      list(model.language_projection.parameters())
    vision_params   = []
    encoder_layers  = model.vision_model.encoder.layers
    for layer in encoder_layers[-VISION_UNFREEZE:]:
        vision_params.extend(layer.parameters())

    # Filter to only trainable params
    qformer_params = [p for p in qformer_params if p.requires_grad]
    vision_params  = [p for p in vision_params  if p.requires_grad]

    return torch.optim.AdamW([
        {"params": qformer_params, "lr": LR_QFORMER},
        {"params": vision_params,  "lr": LR_VISION},
    ])


# ── VRAM helper ───────────────────────────────────────────────────────────────

def vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 3
    return 0.0


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load processor and model
    print("Loading processor …")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Loading model …")
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(device)

    # Freeze everything, then selectively unfreeze
    freeze_all(model)
    unfreeze_qformer(model)
    unfreeze_vision_last_n(model, VISION_UNFREEZE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # Dataset and loader
    print("Loading dataset …")
    dataset = PicsCanLieDataset(ANNOTATIONS, METADATA, IMAGES_ROOT, MAX_SAMPLES)
    print(f"Samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,          # keep 0 for Windows compatibility
        collate_fn=lambda b: collate_fn(b, processor),
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = build_optimizer(model)
    scaler    = GradScaler(enabled=torch.cuda.is_available())

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(**batch)
                loss    = outputs.loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if step % GRAD_ACCUM == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * GRAD_ACCUM  # un-scale for logging

            if step % LOG_EVERY == 0:
                avg_loss = epoch_loss / step
                elapsed  = time.time() - t0
                print(
                    f"Epoch {epoch} | step {step}/{len(loader)} "
                    f"| loss {avg_loss:.4f} "
                    f"| VRAM {vram_gb():.2f} GB "
                    f"| {elapsed:.0f}s elapsed"
                )

        avg_epoch_loss = epoch_loss / len(loader)
        print(f"\nEpoch {epoch} complete — avg loss {avg_epoch_loss:.4f}\n")

        # Save checkpoint
        ckpt_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        print(f"Checkpoint saved → {ckpt_dir}\n")

    print("Training complete.")


if __name__ == "__main__":
    train()
