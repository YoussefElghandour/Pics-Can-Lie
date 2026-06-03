"""
fix_newsclip_inference.py  (Prompt A — steps 2-4, the part computable without
re-running CLIP on the 7264 test images)

What it does, all on already-saved features (no CLIP/AITR retraining):
  1. Rebuilds the 9 scalars in the TRAINING order
     [clip_prob, clip_sim, deberta, s2, s3, s4, s5, s6, wiki].
  2. Refits StandardScaler on the TRAIN split ONLY (fixes the fit-before-split
     leak) and persists it to fusion_aitr/scalar_scaler.joblib  — this is the
     scaler the fixed test path will load.
  3. Loads the trained AITR, scales scalars with the train-only scaler, and
     produces fused probabilities for the train and internal-val splits.
  4. Freezes a decision threshold on the TRAIN split (max balanced accuracy),
     saves it to fusion_aitr/frozen_threshold.json, and applies it BLIND to val.
  5. Reports honest VAL numbers: AUC + accuracy at the frozen threshold, each
     with a 1000-resample 95% bootstrap CI. No threshold is ever chosen on the
     rows it is scored on.

The TEST set has no saved CLIP embeddings, so its fused_prob must be recomputed
on the 4060 with run_test_inference_fixed.py (this script prepares the scaler +
frozen threshold that runner consumes).
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import config

PROJECT_ROOT  = str(config.ROOT)
CLIP_FEATURES = str(config.CLIP_VAL_FEATURES)
FUSION_SAVE   = str(config.MODELS / "fusion_aitr")
AITR_CKPT     = str(config.AITR_CKPT)
SCALER_OUT    = str(config.SCALER_PATH)
THRESH_OUT    = str(config.THRESH_PATH)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Canonical scalar order — MUST match training (day2_aitr_fusion.ipynb cell 2).
SCALAR_ORDER = ["clip_prob", "clip_sim", "deberta", "s2", "s3", "s4", "s5", "s6", "wiki"]


class AITR(nn.Module):
    """Architecture matching fusion_aitr/aitr_weights.pt (see run_test_inference.py)."""
    def __init__(self, embed_dim=768, scalar_dim=9, num_heads=8, num_layers=2,
                 dropout=0.3, hidden_dim=256):
        super().__init__()
        self.scalar_proj = nn.Sequential(nn.Linear(scalar_dim, embed_dim),
                                          nn.LayerNorm(embed_dim))
        self.type_embedding = nn.Embedding(5, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, img_emb, txt_emb, scalar):
        B = img_emb.size(0)
        scalar_emb = self.scalar_proj(scalar)
        tokens = torch.stack(
            [img_emb, txt_emb, img_emb * txt_emb, img_emb - txt_emb, scalar_emb], dim=1)
        type_ids = torch.arange(5, device=tokens.device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.type_embedding(type_ids)
        cls = self.cls_token.expand(B, -1, -1)
        out = self.transformer(torch.cat([cls, tokens], dim=1))
        return torch.sigmoid(self.classifier(out[:, 0])).squeeze(-1)


def build_raw_scalars(id_df):
    ids = id_df["id"].astype(str)
    clip_probs = np.load(os.path.join(CLIP_FEATURES, "clip_finetuned_probs.npy"))
    clip_sims  = np.load(os.path.join(CLIP_FEATURES, "clip_finetuned_sims.npy"))

    deb = pd.read_csv(str(config.DEBERTA_V2_CSV))
    deb["id"] = deb["id"].astype(str)
    deb_lookup = dict(zip(deb["id"], deb["entailment_score"]))
    deb_scores = np.array([deb_lookup.get(i, 0.33) for i in ids])

    ev = pd.read_csv(str(config.EVIDENCE_CSV))
    ev["id"] = ev["id"].astype(str)
    ev_lookup = {row["id"]: row for _, row in ev.iterrows()}
    s = {k: np.array([ev_lookup.get(i, {}).get(k, 0.0) for i in ids]) for k in ["s2","s3","s4","s5","s6"]}

    wiki = pd.read_csv(str(config.WIKI_NLI_CSV))
    wiki["id"] = wiki["id"].astype(str)
    wiki_lookup = {row["id"]: row for _, row in wiki.iterrows()}
    w1 = np.array([wiki_lookup.get(i, {}).get("wiki_score", 0.33) for i in ids])

    raw = np.column_stack([clip_probs, clip_sims, deb_scores,
                           s["s2"], s["s3"], s["s4"], s["s5"], s["s6"], w1]).astype(np.float32)
    return raw


def balanced_acc(y, pred):
    tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else 0.0
    tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else 0.0
    return 0.5 * (tpr + tnr)


def auc(y, p):
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_ci(y, p, thr, n=1000):
    rng = np.random.default_rng(SEED)
    accs, aucs = [], []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        yb, pb = y[b], p[b]
        if len(np.unique(yb)) < 2:
            continue
        accs.append(((pb >= thr).astype(int) == yb).mean())
        aucs.append(auc(yb, pb))
    pct = lambda a: (round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4))
    return pct(accs), pct(aucs)


def main():
    print(f"Device: {DEVICE}")
    id_df = pd.read_csv(os.path.join(CLIP_FEATURES, "val_sample_ids.csv"))
    id_df["id"] = id_df["id"].astype(str)
    labels = id_df["label"].values.astype(int)

    img = F.normalize(torch.load(os.path.join(CLIP_FEATURES, "clip_img_features.pt")), dim=-1).float()
    txt = F.normalize(torch.load(os.path.join(CLIP_FEATURES, "clip_txt_features.pt")), dim=-1).float()
    raw = build_raw_scalars(id_df)
    print(f"Loaded {len(labels)} samples; scalar order = {SCALAR_ORDER}")

    idx = np.arange(len(labels))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=labels)

    # (2) Scaler fit on TRAIN rows only — fixes the fit-before-split leak.
    scaler = StandardScaler().fit(raw[tr_idx])
    joblib.dump({"scaler": scaler, "order": SCALAR_ORDER,
                 "train_means": raw[tr_idx].mean(0).tolist()}, SCALER_OUT)
    print(f"[scaler] fit on {len(tr_idx)} train rows -> saved {SCALER_OUT}")

    scaled = scaler.transform(raw).astype(np.float32)

    # (3) Load AITR and score.
    model = AITR().to(DEVICE).eval()
    state = torch.load(AITR_CKPT, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)

    @torch.no_grad()
    def score(rows):
        out = np.empty(len(rows), dtype=np.float32)
        for s in range(0, len(rows), 256):
            e = min(s + 256, len(rows)); b = rows[s:e]
            out[s:e] = model(img[b].to(DEVICE), txt[b].to(DEVICE),
                             torch.tensor(scaled[b], device=DEVICE)).cpu().numpy()
        return out

    p_tr = score(tr_idx); y_tr = labels[tr_idx]
    p_va = score(va_idx); y_va = labels[va_idx]

    # (4) Freeze threshold on TRAIN (max balanced accuracy), apply blind to val.
    grid = np.linspace(p_tr.min(), p_tr.max(), 501)
    bals = [balanced_acc(y_tr, (p_tr >= t).astype(int)) for t in grid]
    thr = float(grid[int(np.argmax(bals))])
    json.dump({"frozen_threshold": thr, "selected_on": "train split (max balanced accuracy)",
               "scaler": os.path.basename(SCALER_OUT), "seed": SEED},
              open(THRESH_OUT, "w"), indent=2)
    print(f"[threshold] frozen on train = {thr:.4f} -> saved {THRESH_OUT}")

    # Honest VAL metrics.
    va_acc = float(((p_va >= thr).astype(int) == y_va).mean())
    va_auc = float(auc(y_va, p_va))
    (acc_lo, acc_hi), (auc_lo, auc_hi) = bootstrap_ci(y_va, p_va, thr)

    print("\n=== HONEST VAL (internal val split, threshold frozen on train) ===")
    print(f"  fused_prob range: min={p_va.min():.4f} max={p_va.max():.4f} "
          f"mean={p_va.mean():.4f} std={p_va.std():.4f}")
    print(f"  AUC      = {va_auc:.4f}   95% CI [{auc_lo}, {auc_hi}]")
    print(f"  Accuracy = {va_acc:.4f}   95% CI [{acc_lo}, {acc_hi}]   @thr={thr:.4f}")
    print(f"  (reported notebook val acc ~0.883 was at a val-swept 0.54 threshold)")
    print("\nNote: TEST fused_prob needs the 4060 CLIP pass — run run_test_inference_fixed.py,")
    print("which loads scalar_scaler.joblib + frozen_threshold.json produced here.")


if __name__ == "__main__":
    main()
