"""
Smoke test for 1.2 — compute_clip_features_mmfb.py
---------------------------------------------------
Runs the exact same CLIP + fine-tuned classifier pipeline as 1.2, but on a
stratified 200-sample subset of MMFakeBench_val (so all 4 fake_cls categories
are represented), then prints sanity statistics:

  1. clip_probs:  min/max/mean/std + 10-bin histogram
  2. clip_sims:   min/max/mean/std
  3. Per-gt_answers breakdown (mean clip_prob / clip_sim for Fake vs True),
     plus per-(fake_cls × gt_answers) breakdown so we see whether mismatch /
     visual_vd actually have lower clip_sim than originals.
  4. Tensor shapes and dtypes of saved features.
  5. First 5 rows of the saved sample_ids.csv with their clip_probs.

Stratified by fake_cls — roughly:
  original                    60
  textual_veracity_distortion 60
  mismatch                    60
  visual_veracity_distortion  20

Output dir: mmfakebench_training/smoke_features/
"""


from __future__ import annotations
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Reuse the production module so the pipeline is identical
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_clip_features_mmfb import (  # noqa: E402
    CLIP_CKPT, DEVICE, MMFBDataset, load_finetuned_clip,
)

MMFB_VAL_JSON = Path(_os.path.join(str(_cfg.MMFB_ROOT), 'MMFakeBench_val.json'))
MMFB_VAL_IMG  = Path(_os.path.join(str(_cfg.MMFB_ROOT), 'MMFakeBench_val'))
OUT_DIR       = Path(_os.path.join(str(_cfg.ROOT), 'experiments', 'mmfakebench_training', 'smoke_features'))

QUOTAS = {
    "original":                    60,
    "textual_veracity_distortion": 60,
    "mismatch":                    60,
    "visual_veracity_distortion":  20,
}
SEED = 42
BATCH_SIZE = 64


def stratified_subset(ann):
    rng = np.random.default_rng(SEED)
    by_cat: dict[str, list] = {}
    for s in ann:
        by_cat.setdefault(s.get("fake_cls", ""), []).append(s)
    chosen = []
    for cat, quota in QUOTAS.items():
        pool = by_cat.get(cat, [])
        if len(pool) < quota:
            print(f"[smoke] WARNING: cat={cat} pool={len(pool)} < quota={quota} — taking all")
            chosen.extend(pool)
        else:
            idx = rng.choice(len(pool), size=quota, replace=False)
            chosen.extend([pool[i] for i in idx])
    print(f"[smoke] subset size = {len(chosen)}  "
          f"breakdown = {dict(Counter(s['fake_cls'] for s in chosen))}")
    return chosen


@torch.no_grad()
def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MMFB_VAL_JSON, encoding="utf-8") as f:
        ann_full = json.load(f)
    ann = stratified_subset(ann_full)

    print(f"[smoke] device = {DEVICE}")
    model, preprocess, head = load_finetuned_clip(CLIP_CKPT, DEVICE)

    ds = MMFBDataset(ann, MMFB_VAL_IMG, preprocess)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=0, pin_memory=False, drop_last=False)

    n = len(ds)
    img_feats = torch.empty(n, 768, dtype=torch.float32)
    txt_feats = torch.empty(n, 768, dtype=torch.float32)
    probs = np.empty(n, dtype=np.float32)
    sims  = np.empty(n, dtype=np.float32)

    for idx, img, tok in tqdm(dl, desc="smoke CLIP", unit="batch"):
        img = img.to(DEVICE); tok = tok.to(DEVICE)
        f_img = model.encode_image(img).float()
        f_txt = model.encode_text(tok).float()
        f_img_n = F.normalize(f_img, dim=-1)
        f_txt_n = F.normalize(f_txt, dim=-1)
        cos = (f_img_n * f_txt_n).sum(dim=-1)
        head_in = torch.cat([f_img_n, f_txt_n, cos.unsqueeze(-1)], dim=-1)
        p = torch.sigmoid(head(head_in))

        idx_np = idx.numpy()
        img_feats[idx_np] = f_img.cpu()
        txt_feats[idx_np] = f_txt.cpu()
        probs[idx_np] = p.cpu().numpy()
        sims[idx_np]  = cos.cpu().numpy()

    # ── Save (so user can poke at it later) ─────────────────────────────────
    torch.save(img_feats, OUT_DIR / "clip_img.pt")
    torch.save(txt_feats, OUT_DIR / "clip_txt.pt")
    np.save(OUT_DIR / "clip_probs.npy", probs)
    np.save(OUT_DIR / "clip_sims.npy",  sims)
    df = pd.DataFrame({
        "idx": np.arange(n),
        "sample_id": [f"smoke_{i:04d}" for i in range(n)],
        "image_path":   [s["image_path"]            for s in ann],
        "fake_cls":     [s.get("fake_cls", "")      for s in ann],
        "gt_answers":   [s.get("gt_answers", "")    for s in ann],
        "label":        [(1 if s.get("gt_answers") == "Fake" else 0) for s in ann],
        "clip_prob":    probs,
        "clip_sim":     sims,
    })
    df.to_csv(OUT_DIR / "sample_ids.csv", index=False)

    # ── Sanity prints ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLIP smoke-test sanity report")
    print("=" * 60)

    print("\n[1] clip_probs distribution")
    print(f"    min={probs.min():.4f}  max={probs.max():.4f}  "
          f"mean={probs.mean():.4f}  std={probs.std():.4f}")
    hist, edges = np.histogram(probs, bins=10, range=(0.0, 1.0))
    for i, h in enumerate(hist):
        bar = "#" * int(40 * h / max(hist.max(), 1))
        print(f"    [{edges[i]:.2f}-{edges[i+1]:.2f}]  {h:4d}  {bar}")

    print("\n[2] clip_sims distribution")
    print(f"    min={sims.min():.4f}  max={sims.max():.4f}  "
          f"mean={sims.mean():.4f}  std={sims.std():.4f}")

    print("\n[3a] mean by gt_answers")
    for gt in ["Fake", "True"]:
        m = df["gt_answers"] == gt
        if m.any():
            print(f"    gt={gt:<5} n={int(m.sum()):<4}  "
                  f"clip_prob_mean={df.loc[m,'clip_prob'].mean():.4f}  "
                  f"clip_sim_mean={df.loc[m,'clip_sim'].mean():.4f}")

    print("\n[3b] mean by (fake_cls, gt_answers)")
    for cat in QUOTAS.keys():
        for gt in ["Fake", "True"]:
            m = (df["fake_cls"] == cat) & (df["gt_answers"] == gt)
            if m.any():
                print(f"    {cat:<32} gt={gt:<5} n={int(m.sum()):<3}  "
                      f"prob={df.loc[m,'clip_prob'].mean():.4f}  "
                      f"sim={df.loc[m,'clip_sim'].mean():.4f}")

    print("\n[4] saved tensor shapes / dtypes")
    print(f"    clip_img.pt   shape={tuple(img_feats.shape)}  dtype={img_feats.dtype}  "
          f"|x|_mean={img_feats.norm(dim=-1).mean().item():.2f}")
    print(f"    clip_txt.pt   shape={tuple(txt_feats.shape)}  dtype={txt_feats.dtype}  "
          f"|x|_mean={txt_feats.norm(dim=-1).mean().item():.2f}")
    print(f"    clip_probs    shape={probs.shape}  dtype={probs.dtype}")
    print(f"    clip_sims     shape={sims.shape}   dtype={sims.dtype}")

    print("\n[5] first 5 rows of sample_ids.csv")
    cols = ["idx", "sample_id", "fake_cls", "gt_answers", "label", "clip_prob", "clip_sim"]
    print(df[cols].head(5).to_string(index=False))

    print(f"\nSaved to {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run()
