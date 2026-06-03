"""
1.4 — train_aitr_mmfb.py
------------------------
Retrain the AITR transformer fusion model on MMFakeBench's *train* split
(test.json / 10000 samples), keeping the architecture identical to the
NewsCLIPpings-trained AITR and warm-starting from its weights.

Scalars used here (3-dim — only signals we have for MMFakeBench):
    [clip_prob, clip_sim, ateeq_score_ft]

Architecture: 5 tokens × 768
  1. img_emb  (L2-normalized CLIP image feature)
  2. txt_emb  (L2-normalized CLIP text feature)
  3. img * txt
  4. img - txt
  5. scalar_proj(scalars)
+ learnable CLS token → 2-layer Transformer (8h, dim_ff=1536, pre-norm, GELU)
→ classifier LN→Linear(768→256)→GELU→Dropout→Linear(256→1).

Warm-start: load NewsCLIPpings aitr_weights.pt; reinitialise ONLY
scalar_proj (because 9 → 3 scalars). Everything else inherits.

Training:
  - 80/20 split (random_state=42) of train split for train / internal-val
  - BCEWithLogitsLoss with pos_weight = n_real / n_fake (train fold only)
  - AdamW: lr=1e-4 for scalar_proj, lr=1e-5 for the rest
  - Batch 64, up to 20 epochs, early stop on internal-val F1 (patience=4)
  - 1-epoch linear warmup → cosine decay
  - num_workers=0, pin_memory=False  (Windows)

Outputs:
  mmfakebench_training/aitr_mmfb_best.pt
  mmfakebench_training/training_log.csv

Run:
    python mmfakebench_training/train_aitr_mmfb.py
"""


from __future__ import annotations
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(str(_cfg.ROOT))
OUT_ROOT     = PROJECT_ROOT / "mmfakebench_training"
TRAIN_FEAT   = OUT_ROOT / "train_features"
ATEEQ_CSV    = OUT_ROOT / "train_ateeq_scores.csv"
AITR_INIT    = PROJECT_ROOT / "fusion_aitr" / "aitr_weights.pt"

BEST_PATH    = OUT_ROOT / "aitr_mmfb_best.pt"
LOG_PATH     = OUT_ROOT / "training_log.csv"

# ── Hyperparams ──────────────────────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 64
EPOCHS       = 20
PATIENCE     = 4
LR_NEW       = 1e-4      # for scalar_proj (3-dim → 768 — new shape)
LR_WARM      = 1e-5      # for everything inherited from NewsCLIPpings AITR
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── AITR architecture — must match fusion_aitr/aitr_weights.pt ──────────────
class AITR(nn.Module):
    def __init__(self, embed_dim: int = 768, scalar_dim: int = 3,
                 num_heads: int = 8, num_layers: int = 2,
                 dropout: float = 0.3, hidden_dim: int = 256):
        super().__init__()
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.type_embedding = nn.Embedding(5, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,   # 1536  (matches checkpoint)
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        B = img_emb.size(0)
        prod_emb = img_emb * txt_emb
        diff_emb = img_emb - txt_emb
        scalar_emb = self.scalar_proj(scalar)

        tokens = torch.stack([img_emb, txt_emb, prod_emb, diff_emb, scalar_emb], dim=1)  # (B,5,768)
        type_ids = torch.arange(5, device=tokens.device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.type_embedding(type_ids)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)                 # (B,6,768)
        out = self.transformer(tokens)
        return self.classifier(out[:, 0]).squeeze(-1)             # (B,)


def warm_start(model: AITR, ckpt_path: Path) -> None:
    """Load NewsCLIPpings AITR weights, skipping scalar_proj (9→3 mismatch)."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    own = model.state_dict()
    loaded, skipped = 0, []
    for k, v in sd.items():
        if k.startswith("scalar_proj."):
            skipped.append(k)
            continue
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded += 1
        else:
            skipped.append(k)
    model.load_state_dict(own)
    print(f"[warm-start] loaded {loaded} tensors from {ckpt_path.name}; "
          f"skipped {len(skipped)} (incl. scalar_proj): {skipped[:3]}")


# ── Dataset built from saved tensors ────────────────────────────────────────
class AITRFeatureDataset(Dataset):
    def __init__(self, img: torch.Tensor, txt: torch.Tensor,
                 probs: np.ndarray, sims: np.ndarray,
                 ateeq: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.img = img
        self.txt = txt
        self.probs = probs
        self.sims = sims
        self.ateeq = ateeq
        self.labels = labels
        self.indices = indices  # rows into the underlying full arrays

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img = F.normalize(self.img[idx], dim=-1)
        txt = F.normalize(self.txt[idx], dim=-1)
        scl = torch.tensor(
            [self.probs[idx], self.sims[idx], self.ateeq[idx]],
            dtype=torch.float32,
        )
        y = float(self.labels[idx])
        return img, txt, scl, torch.tensor(y, dtype=torch.float32)


# ── Data loading & alignment by sample_id ───────────────────────────────────
def load_train_data():
    sample_ids = pd.read_csv(TRAIN_FEAT / "sample_ids.csv")   # idx,sample_id,...,label
    img = torch.load(TRAIN_FEAT / "clip_img.pt", map_location="cpu", weights_only=False)
    txt = torch.load(TRAIN_FEAT / "clip_txt.pt", map_location="cpu", weights_only=False)
    probs = np.load(TRAIN_FEAT / "clip_probs.npy")
    sims  = np.load(TRAIN_FEAT / "clip_sims.npy")

    ateeq_df = pd.read_csv(ATEEQ_CSV)                         # has sample_id, ateeq_score_ft
    # Strict alignment by sample_id (never by row position)
    merged = sample_ids.merge(
        ateeq_df[["sample_id", "ateeq_score_ft"]],
        on="sample_id", how="left", validate="one_to_one",
    )
    if merged["ateeq_score_ft"].isna().any():
        n_miss = int(merged["ateeq_score_ft"].isna().sum())
        raise RuntimeError(f"{n_miss} train rows missing ateeq_score_ft — re-run 1.3")
    # Clip any -1 sentinels that survived to median
    bad = merged["ateeq_score_ft"] < 0
    if bad.any():
        med = float(merged.loc[~bad, "ateeq_score_ft"].median())
        print(f"[data] {int(bad.sum())} ateeq sentinel(-1) rows -> median {med:.3f}")
        merged.loc[bad, "ateeq_score_ft"] = med

    ateeq = merged["ateeq_score_ft"].to_numpy(dtype=np.float32)
    labels = merged["label"].to_numpy(dtype=np.int64)

    assert len(ateeq) == len(img) == len(txt) == len(probs) == len(sims) == len(labels)
    return img, txt, probs, sims, ateeq, labels


def epoch_loop(model, dl, optim, loss_fn, scheduler=None, train: bool = True, desc: str = ""):
    model.train(train)
    losses, all_probs, all_labels = [], [], []
    for img, txt, scl, y in tqdm(dl, desc=desc, leave=False, unit="batch"):
        img = img.to(DEVICE); txt = txt.to(DEVICE); scl = scl.to(DEVICE); y = y.to(DEVICE)
        if train:
            optim.zero_grad()
        logits = model(img, txt, scl)
        loss = loss_fn(logits, y)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
        losses.append(loss.item() * y.size(0))
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())
    if scheduler is not None and train:
        scheduler.step()

    n = sum(1 for _ in all_labels)
    avg_loss = float(np.sum(losses) / max(n, 1))
    probs_arr = np.asarray(all_probs)
    labels_arr = np.asarray(all_labels).astype(int)
    preds = (probs_arr > 0.5).astype(int)
    metrics = {
        "loss": avg_loss,
        "acc":  float(accuracy_score(labels_arr, preds)),
        "f1":   float(f1_score(labels_arr, preds, zero_division=0)),
        "auc":  float(roc_auc_score(labels_arr, probs_arr)) if len(set(labels_arr)) > 1 else float("nan"),
    }
    return metrics


def main() -> None:
    print(f"Device: {DEVICE}")
    img, txt, probs, sims, ateeq, labels = load_train_data()
    n = len(labels)
    print(f"Loaded {n} train samples — fake={labels.sum()} real={(labels==0).sum()}")

    idx_all = np.arange(n)
    tr_idx, va_idx = train_test_split(
        idx_all, test_size=0.2, random_state=SEED, stratify=labels,
    )
    n_fake_tr = int(labels[tr_idx].sum())
    n_real_tr = int((labels[tr_idx] == 0).sum())
    pos_weight_val = n_real_tr / max(n_fake_tr, 1)
    print(f"Train fold: fake={n_fake_tr} real={n_real_tr}  pos_weight={pos_weight_val:.4f}")

    tr_ds = AITRFeatureDataset(img, txt, probs, sims, ateeq, labels, tr_idx)
    va_ds = AITRFeatureDataset(img, txt, probs, sims, ateeq, labels, va_idx)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, pin_memory=False)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=0, pin_memory=False)

    # ── Model + warm start ───────────────────────────────────────────────────
    model = AITR(scalar_dim=3).to(DEVICE)
    warm_start(model, AITR_INIT)

    # ── Param groups: fresh scalar_proj gets a higher LR ─────────────────────
    new_params, warm_params = [], []
    for name, p in model.named_parameters():
        if name.startswith("scalar_proj."):
            new_params.append(p)
        else:
            warm_params.append(p)
    optim = torch.optim.AdamW(
        [{"params": new_params, "lr": LR_NEW},
         {"params": warm_params, "lr": LR_WARM}],
        weight_decay=WEIGHT_DECAY,
    )

    # 1-epoch linear warmup + cosine over remaining epochs (per param group)
    def lr_lambda(epoch: int) -> float:
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / max(WARMUP_EPOCHS, 1)
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    pos_w = torch.tensor([pos_weight_val], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # ── Train loop ───────────────────────────────────────────────────────────
    best_f1, best_epoch, stale = -1.0, -1, 0
    log_rows = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        tr = epoch_loop(model, tr_dl, optim, loss_fn, scheduler, train=True,
                        desc=f"epoch {epoch+1}/{EPOCHS} [train]")
        with torch.no_grad():
            va = epoch_loop(model, va_dl, optim, loss_fn, scheduler=None, train=False,
                            desc=f"epoch {epoch+1}/{EPOCHS} [val]  ")
        lr_now = optim.param_groups[0]["lr"]
        print(f"  ep {epoch+1:02d} | "
              f"tr loss={tr['loss']:.4f} f1={tr['f1']:.4f} acc={tr['acc']:.4f} | "
              f"va loss={va['loss']:.4f} f1={va['f1']:.4f} acc={va['acc']:.4f} auc={va['auc']:.4f} | "
              f"lr={lr_now:.2e}")
        log_rows.append({
            "epoch": epoch + 1,
            "lr_new": optim.param_groups[0]["lr"],
            "lr_warm": optim.param_groups[1]["lr"],
            "train_loss": tr["loss"], "train_f1": tr["f1"], "train_acc": tr["acc"],
            "val_loss":   va["loss"], "val_f1":   va["f1"], "val_acc":   va["acc"],
            "val_auc":    va["auc"],
        })

        if va["f1"] > best_f1:
            best_f1, best_epoch, stale = va["f1"], epoch + 1, 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_metrics": va,
                "pos_weight": pos_weight_val,
                "scalar_dim": 3,
            }, BEST_PATH)
            print(f"  -> new best F1={best_f1:.4f}  saved {BEST_PATH.name}")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"  early stop (no improvement for {PATIENCE} epochs)")
                break

    pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
    print(f"\nDone in {time.time()-t0:.1f}s.  best F1={best_f1:.4f} @ epoch {best_epoch}")
    print(f"Best checkpoint -> {BEST_PATH}")
    print(f"Training log    -> {LOG_PATH}")


if __name__ == "__main__":
    main()
