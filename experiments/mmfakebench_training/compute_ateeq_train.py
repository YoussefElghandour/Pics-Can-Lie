"""
1.3 — compute_ateeq_train.py
----------------------------
Run the fine-tuned Ateeq (SiglipForImageClassification) on every MMFakeBench
image in both splits and save per-sample 'ai' probabilities.

Outputs:
  mmfakebench_training/train_ateeq_scores.csv  (10000 rows)
  mmfakebench_training/val_ateeq_scores_full.csv  (1000 rows — fills in the 800
                                                   that mmfakebench_ai_scores_finetuned.csv
                                                   didn't cover)

CSV columns: idx, sample_id, image_path, fake_cls, gt_answers, label, ateeq_score_ft

Re-runnable: writes a checkpoint every CHECKPOINT_EVERY samples to
  mmfakebench_training/_ateeq_ckpt_<split>.csv  and resumes from it on restart.

Windows rules: num_workers=0, pin_memory=False.
"""


from __future__ import annotations
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(str(_cfg.ROOT))
MMFB_ROOT    = Path(_os.path.join(str(_cfg.MMFB_ROOT)))
ATEEQ_DIR    = PROJECT_ROOT / "ai_detector_finetuned"
OUT_ROOT     = PROJECT_ROOT / "mmfakebench_training"

SPLITS = {
    "train": {
        "json": MMFB_ROOT / "MMFakeBench_test.json",
        "img_root": MMFB_ROOT / "MMFakeBench_test",
        "out_csv": OUT_ROOT / "train_ateeq_scores.csv",
    },
    "val": {
        "json": MMFB_ROOT / "MMFakeBench_val.json",
        "img_root": MMFB_ROOT / "MMFakeBench_val",
        "out_csv": OUT_ROOT / "val_ateeq_scores_full.csv",
    },
}

BATCH_SIZE        = 16
CHECKPOINT_EVERY  = 200   # samples
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset: image only ──────────────────────────────────────────────────────
class ImagesOnly(Dataset):
    def __init__(self, ann: List[dict], img_root: Path, processor):
        self.ann = ann
        self.img_root = img_root
        self.processor = processor

    def __len__(self) -> int:
        return len(self.ann)

    def __getitem__(self, idx: int):
        rel = self.ann[idx]["image_path"].lstrip("/\\")
        path = self.img_root / rel
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        proc = self.processor(images=img, return_tensors="pt")
        # processor returns (1, C, H, W); strip batch dim
        return idx, proc["pixel_values"].squeeze(0)


def run_split(name: str, model, processor, ai_idx: int, device: str) -> None:
    cfg = SPLITS[name]
    out_csv: Path = cfg["out_csv"]
    ckpt_csv = OUT_ROOT / f"_ateeq_ckpt_{name}.csv"

    with open(cfg["json"], encoding="utf-8") as f:
        ann = json.load(f)
    n = len(ann)
    print(f"\n[{name}] {n} samples")

    # ── Resume from checkpoint if present ────────────────────────────────────
    scores = np.full(n, np.nan, dtype=np.float32)
    done_count = 0
    if ckpt_csv.exists():
        prev = pd.read_csv(ckpt_csv)
        mask = ~prev["ateeq_score_ft"].isna()
        scores[prev.loc[mask, "idx"].to_numpy()] = prev.loc[mask, "ateeq_score_ft"].to_numpy(dtype=np.float32)
        done_count = int(mask.sum())
        print(f"[{name}] resumed: {done_count}/{n} already scored")

    # Build a dataset over the still-pending indices only
    pending_idx = np.where(np.isnan(scores))[0].tolist()
    if not pending_idx:
        print(f"[{name}] nothing to do — already complete")
    else:
        pending_ann = [ann[i] for i in pending_idx]
        # remap dataset idx -> original idx via a lookup list
        ds = ImagesOnly(pending_ann, cfg["img_root"], processor)
        dl = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        since_ckpt = 0
        with torch.no_grad():
            for ds_idx, px in tqdm(dl, desc=f"[{name}] Ateeq", unit="batch"):
                px = px.to(device, non_blocking=False)
                logits = model(pixel_values=px).logits
                probs = torch.softmax(logits, dim=-1)[:, ai_idx].cpu().numpy()
                ds_idx_np = ds_idx.numpy()
                orig_idx = np.asarray([pending_idx[i] for i in ds_idx_np])
                scores[orig_idx] = probs
                since_ckpt += len(orig_idx)
                if since_ckpt >= CHECKPOINT_EVERY:
                    _write_csv(ckpt_csv, ann, scores, name)
                    since_ckpt = 0

    # ── Final write (full CSV, no NaNs expected) ─────────────────────────────
    if np.isnan(scores).any():
        n_miss = int(np.isnan(scores).sum())
        print(f"[{name}] WARNING: {n_miss} scores still NaN — leaving as -1")
        scores = np.where(np.isnan(scores), -1.0, scores)

    _write_csv(out_csv, ann, scores, name)
    if ckpt_csv.exists():
        ckpt_csv.unlink()
    print(f"[{name}] saved -> {out_csv}  ({n} rows)")
    valid = scores[scores >= 0]
    if len(valid):
        print(f"        ateeq_score_ft mean={valid.mean():.3f}  median={np.median(valid):.3f}")


def _write_csv(path: Path, ann: List[dict], scores: np.ndarray, name: str) -> None:
    n = len(ann)
    # NOTE: name is passed explicitly — earlier heuristic
    # `"train" in str(path).lower()` matched the substring inside
    # "mmfakebench_training" for val paths and mislabeled rows as "train_*".
    df = pd.DataFrame({
        "idx": np.arange(n),
        "sample_id": [f"{name}_{i:06d}" for i in range(n)],
        "image_path": [s["image_path"] for s in ann],
        "fake_cls": [s.get("fake_cls", "") for s in ann],
        "gt_answers": [s.get("gt_answers", "") for s in ann],
        "label": [(1 if s.get("gt_answers") == "Fake" else 0) for s in ann],
        "ateeq_score_ft": scores,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Loading fine-tuned Ateeq from {ATEEQ_DIR}")
    # The save dir has its own config + preprocessor_config — load directly
    processor = AutoImageProcessor.from_pretrained(str(ATEEQ_DIR))
    model = AutoModelForImageClassification.from_pretrained(str(ATEEQ_DIR)).to(DEVICE)
    model.eval()

    id2label = model.config.id2label
    label2id = {v.lower(): int(k) for k, v in id2label.items()}
    ai_idx = label2id.get("ai", 0)
    print(f"id2label={id2label}  ai_idx={ai_idx}")

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for s in splits:
        run_split(s, model, processor, ai_idx, DEVICE)
    print("\nDone.")


if __name__ == "__main__":
    main()
