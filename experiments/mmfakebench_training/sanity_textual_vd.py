"""
Sanity check on the textual_veracity_distortion (textual_vd) lift.

Two questions:
  (1) Is clip_prob genuinely discriminating textual_vd vs original on val?
      A real signal => mean(clip_prob | textual_vd) - mean(clip_prob | original) > 0.10
      An artifact   => means are nearly identical and the lift is just threshold
                       positioning under 70% class prior.
  (2) Of the 61 textual_vd false negatives at thr=0.30, are their AITR probs
      flat (genuinely indistinguishable) or bimodal (signal we're throwing away
      to threshold choice)?

Outputs: prints to stdout, no files written.
"""

from __future__ import annotations
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_aitr_mmfb import AITR  # noqa: E402

ROOT     = Path(_os.path.join(str(_cfg.ROOT), 'experiments', 'mmfakebench_training'))
VAL_FEAT = ROOT / "val_features"
ATEEQ    = ROOT / "val_ateeq_scores_full.csv"
CKPT     = ROOT / "aitr_mmfb_best.pt"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
BEST_T   = 0.30

# ── Load val ─────────────────────────────────────────────────────────────────
sid   = pd.read_csv(VAL_FEAT / "sample_ids.csv")
img   = torch.load(VAL_FEAT / "clip_img.pt", map_location="cpu", weights_only=False)
txt   = torch.load(VAL_FEAT / "clip_txt.pt", map_location="cpu", weights_only=False)
probs = np.load(VAL_FEAT / "clip_probs.npy").astype(np.float32)
sims  = np.load(VAL_FEAT / "clip_sims.npy").astype(np.float32)
ateeq = pd.read_csv(ATEEQ)
merged = sid.merge(ateeq[["sample_id","ateeq_score_ft"]], on="sample_id", validate="one_to_one")
ateeq_arr = merged["ateeq_score_ft"].to_numpy(dtype=np.float32)
labels    = merged["label"].to_numpy(dtype=np.int64)
cats      = merged["fake_cls"].to_numpy()

# ── AITR forward pass on all val ─────────────────────────────────────────────
model = AITR(scalar_dim=3).to(DEVICE)
ckpt  = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

with torch.no_grad():
    p_aitr = np.empty(len(labels), dtype=np.float32)
    for s in range(0, len(labels), 64):
        e = min(s+64, len(labels))
        i_n = F.normalize(img[s:e], dim=-1).to(DEVICE)
        t_n = F.normalize(txt[s:e], dim=-1).to(DEVICE)
        scl = torch.tensor(np.stack([probs[s:e], sims[s:e], ateeq_arr[s:e]], axis=1),
                           dtype=torch.float32, device=DEVICE)
        p_aitr[s:e] = torch.sigmoid(model(i_n, t_n, scl)).cpu().numpy()

# ── (1) clip_prob and clip_sim means by category ────────────────────────────
print("=" * 70)
print("(1) clip_prob / clip_sim signal by (fake_cls)")
print("=" * 70)
def stats(name, mask):
    if not mask.any(): return
    print(f"  {name:<35} n={int(mask.sum()):<4} "
          f"clip_prob mean={probs[mask].mean():.4f} std={probs[mask].std():.4f}  "
          f"clip_sim mean={sims[mask].mean():.4f} std={sims[mask].std():.4f}  "
          f"ateeq mean={ateeq_arr[mask].mean():.4f}")

for c in ["original", "mismatch", "textual_veracity_distortion", "visual_veracity_distortion"]:
    stats(c, cats == c)

# Direct comparison: textual_vd vs original
m_tvd  = (cats == "textual_veracity_distortion")
m_orig = (cats == "original")
diff_prob = probs[m_tvd].mean() - probs[m_orig].mean()
diff_sim  = sims[m_tvd].mean()  - sims[m_orig].mean()
diff_ate  = ateeq_arr[m_tvd].mean() - ateeq_arr[m_orig].mean()
print()
print(f"  Δ(textual_vd − original):  "
      f"clip_prob={diff_prob:+.4f}  clip_sim={diff_sim:+.4f}  ateeq={diff_ate:+.4f}")
print(f"  Verdict (clip_prob):  "
      f"{'REAL SIGNAL (>0.10)' if abs(diff_prob)>0.10 else 'WEAK — likely threshold artifact'}")

# AITR's own probs by category (post-fusion)
print()
print("AITR prob (post-fusion) by category:")
for c in ["original", "mismatch", "textual_veracity_distortion", "visual_veracity_distortion"]:
    m = cats == c
    if m.any():
        print(f"  {c:<35} n={int(m.sum()):<4} "
              f"AITR_prob mean={p_aitr[m].mean():.4f} std={p_aitr[m].std():.4f} "
              f"median={np.median(p_aitr[m]):.4f}")

# ── (2) Confusion + miss-distribution on textual_vd at thr=0.30 ─────────────
print()
print("=" * 70)
print("(2) textual_vd confusion @ thr=0.30  (n=300, all true label=Fake)")
print("=" * 70)
preds_tvd = (p_aitr[m_tvd] > BEST_T).astype(int)
labels_tvd = labels[m_tvd]
TP = int(((preds_tvd == 1) & (labels_tvd == 1)).sum())
FN = int(((preds_tvd == 0) & (labels_tvd == 1)).sum())
print(f"  TP (correctly flagged Fake) = {TP}")
print(f"  FN (missed Fake)            = {FN}")

# AITR-prob distribution for the FN set
miss_probs = p_aitr[m_tvd][preds_tvd == 0]
hit_probs  = p_aitr[m_tvd][preds_tvd == 1]
print()
print(f"  FN AITR prob:  min={miss_probs.min():.4f}  max={miss_probs.max():.4f}  "
      f"mean={miss_probs.mean():.4f}  std={miss_probs.std():.4f}  median={np.median(miss_probs):.4f}")
print(f"  TP AITR prob:  min={hit_probs.min():.4f}  max={hit_probs.max():.4f}  "
      f"mean={hit_probs.mean():.4f}  std={hit_probs.std():.4f}  median={np.median(hit_probs):.4f}")

# Histogram of FN probs (within [0, 0.30])
print()
print("  FN histogram (10 bins over [0.00, 0.30]):")
hist, edges = np.histogram(miss_probs, bins=10, range=(0.0, 0.30))
for i, h in enumerate(hist):
    bar = "#" * int(40 * h / max(hist.max(), 1))
    print(f"    [{edges[i]:.3f}-{edges[i+1]:.3f}]  {h:3d}  {bar}")

# Borderline vs deep-miss
near_t = (miss_probs >= 0.20).sum()
mid    = ((miss_probs >= 0.10) & (miss_probs < 0.20)).sum()
deep   = (miss_probs < 0.10).sum()
print()
print(f"  Borderline (>=0.20, within 0.10 of thr): {int(near_t)}")
print(f"  Mid       (0.10–0.20):                    {int(mid)}")
print(f"  Deep miss (<0.10):                        {int(deep)}")

# Recall at lower thresholds — how many of the FNs would we recover by lowering thr?
print()
print("  textual_vd recall as threshold lowers (overall acc trade-off):")
overall_acc = lambda t: float(((p_aitr > t).astype(int) == labels).mean())
tvd_recall  = lambda t: float(((p_aitr[m_tvd] > t).astype(int) == 1).mean())
for t in [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]:
    print(f"    thr={t:.2f}  tvd_recall={tvd_recall(t)*100:5.2f}%   "
          f"overall_acc={overall_acc(t)*100:5.2f}%")

print()
print("Verdict:")
# Bimodal if both deep and near_t buckets are populated
if near_t >= 0.30 * FN and deep >= 0.30 * FN:
    print("  BIMODAL — many FNs are borderline (>=0.20). Some signal lost to threshold.")
elif near_t >= 0.20 * FN:
    print("  PARTIALLY BIMODAL — meaningful borderline mass; lowering thr trades some real recall.")
else:
    print("  FLAT/DEEP — most FNs are deep below threshold; model genuinely can't see them.")
