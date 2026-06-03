"""
1.2 — compute_clip_features_mmfb.py
-----------------------------------
For both MMFakeBench splits (train = test.json, 10000 samples;
val = val.json, 1000 samples) compute and save:

  - clip_img.pt        (N, 768) float32  — UNNORMALIZED CLIP image features
  - clip_txt.pt        (N, 768) float32  — UNNORMALIZED CLIP text features
  - clip_probs.npy     (N,)     float32  — sigmoid of fine-tuned classifier head
  - clip_sims.npy      (N,)     float32  — cosine sim of L2-normalized features
  - sample_ids.csv     (idx, sample_id, image_path, fake_cls, gt_answers, label)

Using the fine-tuned CLIP ViT-L/14 v2 head at
  D:\\Pics Can Lie\\clip_finetuned_v2\\clip_classifier.pt

Critical Windows + CLIP rules respected:
  - clip.load('ViT-L/14', device=device, jit=False), then .float()  — no .half()
  - DataLoader: num_workers=0, pin_memory=False
  - Features stored UNNORMALIZED (matches v2 convention; the AITR dataset normalizes at use)
  - Label convention: (gt_answers == 'Fake').astype(int)  (NEVER 'False')

Independent splits — re-runnable; pass --split train|val|both.

Run:
    python mmfakebench_training/compute_clip_features_mmfb.py --split both
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
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate odd files in MMFakeBench

try:
    import clip  # OpenAI clip
except ImportError as e:
    sys.exit("openai-clip not installed:  pip install git+https://github.com/openai/CLIP.git")


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(str(_cfg.ROOT))
MMFB_ROOT      = Path(_os.path.join(str(_cfg.MMFB_ROOT)))

CLIP_CKPT      = PROJECT_ROOT / "clip_finetuned_v2" / "clip_classifier.pt"
OUT_ROOT       = PROJECT_ROOT / "mmfakebench_training"

SPLITS = {
    "train": {
        "json": MMFB_ROOT / "MMFakeBench_test.json",
        "img_root": MMFB_ROOT / "MMFakeBench_test",
        "out_dir": OUT_ROOT / "train_features",
    },
    "val": {
        "json": MMFB_ROOT / "MMFakeBench_val.json",
        "img_root": MMFB_ROOT / "MMFakeBench_val",
        "out_dir": OUT_ROOT / "val_features",
    },
}

BATCH_SIZE = 64  # 4060 8GB; drop to 32 on OOM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Fine-tuned CLIP classifier head ──────────────────────────────────────────
# Inferred from clip_classifier.pt:  Linear(1537,512) → BN → ReLU → Dropout
#                                    Linear(512,128)  → BN → ReLU → Dropout
#                                    Linear(128,1)
# Input = concat([img_feat (768), txt_feat (768), cos_sim (1)])
class CLIPClassifierHead(nn.Module):
    def __init__(self, in_dim: int = 1537, hidden1: int = 512, hidden2: int = 128, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden1),     # .0
            nn.BatchNorm1d(hidden1),        # .1
            nn.ReLU(),                      # .2
            nn.Dropout(dropout),            # .3
            nn.Linear(hidden1, hidden2),    # .4
            nn.BatchNorm1d(hidden2),        # .5
            nn.ReLU(),                      # .6
            nn.Dropout(dropout),            # .7
            nn.Linear(hidden2, 1),          # .8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


def load_finetuned_clip(ckpt_path: Path, device: str):
    """Load CLIP ViT-L/14 + fine-tuned weights + classifier head."""
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    model = model.float()  # IMPORTANT on Windows / non-fp16 boxes
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    ms = sd["model_state"]

    clip_sd = {k[len("clip."):]: v for k, v in ms.items() if k.startswith("clip.")}
    missing, unexpected = model.load_state_dict(clip_sd, strict=False)
    if unexpected:
        print(f"[clip] unexpected keys (first 3): {unexpected[:3]}")
    if missing:
        print(f"[clip] missing keys (first 3): {missing[:3]}")
    model.eval()

    head = CLIPClassifierHead().to(device)
    head_sd = {k[len("classifier."):]: v for k, v in ms.items() if k.startswith("classifier.")}
    # the head's submodule is also named .classifier — re-prefix
    head_sd = {f"classifier.{k}": v for k, v in head_sd.items()}
    head.load_state_dict(head_sd, strict=True)
    head.eval()

    print(f"[clip] loaded fine-tuned v2 (epoch={sd.get('epoch','?')})")
    return model, preprocess, head


# ── Dataset ──────────────────────────────────────────────────────────────────
class MMFBDataset(Dataset):
    def __init__(self, annotations: List[dict], img_root: Path, preprocess):
        self.ann = annotations
        self.img_root = img_root
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.ann)

    def __getitem__(self, idx: int):
        s = self.ann[idx]
        rel = s["image_path"].lstrip("/\\")
        path = self.img_root / rel
        try:
            img = Image.open(path).convert("RGB")
            img_t = self.preprocess(img)
        except Exception:
            # Grey placeholder — keeps batch shapes consistent; rare in MMFakeBench
            img_t = self.preprocess(Image.new("RGB", (224, 224), (128, 128, 128)))
        # CLIP tokenizer is at module level
        # context_length=77 default — text > 77 tokens is truncated
        try:
            tok = clip.tokenize([s["text"]], truncate=True)[0]
        except Exception:
            tok = clip.tokenize([""], truncate=True)[0]
        return idx, img_t, tok


# ── Per-split runner ─────────────────────────────────────────────────────────
@torch.no_grad()
def run_split(name: str, model, preprocess, head, device: str) -> None:
    cfg = SPLITS[name]
    json_path: Path = cfg["json"]
    img_root: Path = cfg["img_root"]
    out_dir: Path = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        ann = json.load(f)
    print(f"\n[{name}] {len(ann)} samples from {json_path.name}")

    ds = MMFBDataset(ann, img_root, preprocess)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,        # Windows!
        pin_memory=False,     # Windows!
        drop_last=False,
    )

    n = len(ds)
    img_feats = torch.empty(n, 768, dtype=torch.float32)
    txt_feats = torch.empty(n, 768, dtype=torch.float32)
    probs = np.empty(n, dtype=np.float32)
    sims = np.empty(n, dtype=np.float32)

    for idx, img, tok in tqdm(dl, desc=f"[{name}] CLIP", unit="batch"):
        img = img.to(device, non_blocking=False)
        tok = tok.to(device, non_blocking=False)

        f_img = model.encode_image(img).float()         # (B, 768)
        f_txt = model.encode_text(tok).float()          # (B, 768)

        f_img_n = F.normalize(f_img, dim=-1)
        f_txt_n = F.normalize(f_txt, dim=-1)
        cos = (f_img_n * f_txt_n).sum(dim=-1)           # (B,)

        head_in = torch.cat([f_img_n, f_txt_n, cos.unsqueeze(-1)], dim=-1)  # (B, 1537)
        logits = head(head_in)
        p = torch.sigmoid(logits)                       # (B,)

        idx_np = idx.numpy()
        img_feats[idx_np] = f_img.cpu()
        txt_feats[idx_np] = f_txt.cpu()
        probs[idx_np] = p.cpu().numpy()
        sims[idx_np] = cos.cpu().numpy()

    # ── Save ─────────────────────────────────────────────────────────────────
    torch.save(img_feats, out_dir / "clip_img.pt")
    torch.save(txt_feats, out_dir / "clip_txt.pt")
    np.save(out_dir / "clip_probs.npy", probs)
    np.save(out_dir / "clip_sims.npy", sims)

    df = pd.DataFrame({
        "idx": np.arange(n),
        "sample_id": [f"{name}_{i:06d}" for i in range(n)],
        "image_path": [s["image_path"] for s in ann],
        "text_source": [s.get("text_source", "") for s in ann],
        "image_source": [s.get("image_source", "") for s in ann],
        "fake_cls": [s.get("fake_cls", "") for s in ann],
        "gt_answers": [s.get("gt_answers", "") for s in ann],
        "label": [(1 if s.get("gt_answers") == "Fake" else 0) for s in ann],
    })
    df.to_csv(out_dir / "sample_ids.csv", index=False)
    print(f"[{name}] saved -> {out_dir}")
    print(f"        labels:  fake={df['label'].sum()}  real={(df['label']==0).sum()}")
    print(f"        prob mean={probs.mean():.3f}  sim mean={sims.mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    model, preprocess, head = load_finetuned_clip(CLIP_CKPT, DEVICE)

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for s in splits:
        run_split(s, model, preprocess, head, DEVICE)
    print("\nDone.")


if __name__ == "__main__":
    main()
