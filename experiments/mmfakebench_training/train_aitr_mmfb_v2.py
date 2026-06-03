"""
2.4 — train_aitr_mmfb_v2.py
---------------------------
Train FOUR AITR variants for the Step-2 ablation. Architecture is identical
to Step 1 (5 tokens × 768-d, 8 heads, 2 layers, learnable CLS); only the
scalar set differs:

  A: AITR (2 scalars)       [clip_prob, clip_sim]
  B: AITR + Ateeq           [clip_prob, clip_sim, ateeq_score]       (= Step 1)
  C: AITR + NLI             [clip_prob, clip_sim, deberta_score]
  D: AITR + Ateeq + NLI     [clip_prob, clip_sim, ateeq_score, deberta_score]

Training (identical across variants for fair comparison):
  - Warm-start from fusion_aitr/aitr_weights.pt; reinit ONLY scalar_proj
    (each variant has a different scalar_dim).
  - 80/20 stratified split, random_state=42 — SAME indices for all 4.
  - BCEWithLogitsLoss with pos_weight = n_real/n_fake of the train fold.
  - AdamW: lr=1e-4 on scalar_proj, lr=1e-5 on warm-started weights.
  - Batch 64, up to 20 epochs, patience=4 on internal-val F1.
  - 1-epoch linear warmup + cosine decay (per-variant scheduler).
  - num_workers=0, pin_memory=False (Windows).

Outputs:
  mmfakebench_training/aitr_v2_{A,B,C,D}.pt
  mmfakebench_training/training_log_v2.csv  with rows
      variant,epoch,lr_new,lr_warm,train_loss,val_loss,val_acc,val_f1,val_auc

Run:
    python mmfakebench_training/train_aitr_mmfb_v2.py
"""


from __future__ import annotations
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402

import math
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_aitr_mmfb import AITR, warm_start  # reuse Step-1 model definition

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(str(_cfg.ROOT))
OUT_ROOT     = PROJECT_ROOT / "mmfakebench_training"
TRAIN_FEAT   = OUT_ROOT / "train_features"
VAL_FEAT     = OUT_ROOT / "val_features"
ATEEQ_CSV    = OUT_ROOT / "train_ateeq_scores.csv"
ATEEQ_VAL    = OUT_ROOT / "val_ateeq_scores_full.csv"
DEBERTA_CSV  = OUT_ROOT / "deberta_nli_train.csv"
DEBERTA_VAL  = OUT_ROOT / "deberta_nli_val.csv"
AITR_INIT    = PROJECT_ROOT / "fusion_aitr" / "aitr_weights.pt"
LOG_PATH     = OUT_ROOT / "training_log_v2.csv"

# ── Hyperparams ──────────────────────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 64
EPOCHS       = 20
PATIENCE     = 4
LR_NEW       = 1e-4
LR_WARM      = 1e-5
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Variants ─────────────────────────────────────────────────────────────────
VARIANTS = {
    "A": {"scalars": ["clip_prob", "clip_sim"],                                "ckpt": OUT_ROOT / "aitr_mmfb_v2_A.pt"},
    "B": {"scalars": ["clip_prob", "clip_sim", "ateeq_score"],                 "ckpt": OUT_ROOT / "aitr_mmfb_v2_B.pt"},
    "C": {"scalars": ["clip_prob", "clip_sim", "deberta_score"],               "ckpt": OUT_ROOT / "aitr_mmfb_v2_C.pt"},
    "D": {"scalars": ["clip_prob", "clip_sim", "ateeq_score", "deberta_score"], "ckpt": OUT_ROOT / "aitr_mmfb_v2_D.pt"},
}


# ── Data loading (joined by sample_id) ──────────────────────────────────────
def load_all_train_signals():
    sid = pd.read_csv(TRAIN_FEAT / "sample_ids.csv")
    img = torch.load(TRAIN_FEAT / "clip_img.pt", map_location="cpu", weights_only=False)
    txt = torch.load(TRAIN_FEAT / "clip_txt.pt", map_location="cpu", weights_only=False)
    clip_prob = np.load(TRAIN_FEAT / "clip_probs.npy").astype(np.float32)
    clip_sim  = np.load(TRAIN_FEAT / "clip_sims.npy").astype(np.float32)

    ateeq = pd.read_csv(ATEEQ_CSV)[["sample_id", "ateeq_score_ft"]]
    nli   = pd.read_csv(DEBERTA_CSV)[["sample_id", "deberta_score", "has_evidence"]]

    merged = (
        sid.merge(ateeq, on="sample_id", how="left", validate="one_to_one")
           .merge(nli,   on="sample_id", how="left", validate="one_to_one")
    )
    miss_ate = int(merged["ateeq_score_ft"].isna().sum())
    miss_nli = int(merged["deberta_score"].isna().sum())
    if miss_ate or miss_nli:
        raise RuntimeError(f"missing scalars — ateeq:{miss_ate} nli:{miss_nli}")
    # Replace any -1 sentinel from ateeq with median (Step-1 convention)
    bad = merged["ateeq_score_ft"] < 0
    if bad.any():
        med = float(merged.loc[~bad, "ateeq_score_ft"].median())
        print(f"[data] {int(bad.sum())} ateeq sentinel rows -> median {med:.3f}")
        merged.loc[bad, "ateeq_score_ft"] = med

    signals = {
        "clip_prob":     clip_prob,
        "clip_sim":      clip_sim,
        "ateeq_score":   merged["ateeq_score_ft"].to_numpy(dtype=np.float32),
        "deberta_score": merged["deberta_score"].to_numpy(dtype=np.float32),
    }
    labels = merged["label"].to_numpy(dtype=np.int64)
    return img, txt, signals, labels, merged


# ── Dataset built from saved tensors + a scalar list ────────────────────────
class AITRFeatureDataset(Dataset):
    def __init__(self, img: torch.Tensor, txt: torch.Tensor,
                 scalar_stack: np.ndarray,  # (N, scalar_dim)
                 labels: np.ndarray, indices: np.ndarray):
        self.img = img
        self.txt = txt
        self.scl = scalar_stack
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img = F.normalize(self.img[idx], dim=-1)
        txt = F.normalize(self.txt[idx], dim=-1)
        scl = torch.tensor(self.scl[idx], dtype=torch.float32)
        y = float(self.labels[idx])
        return img, txt, scl, torch.tensor(y, dtype=torch.float32)


# ── Epoch loop ──────────────────────────────────────────────────────────────
def run_epoch(model, dl, optim, loss_fn, train: bool, desc: str):
    model.train(train)
    losses, probs_acc, lab_acc = [], [], []
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
        probs_acc.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        lab_acc.extend(y.detach().cpu().numpy().tolist())
    n = len(lab_acc)
    p = np.asarray(probs_acc); l = np.asarray(lab_acc).astype(int)
    preds = (p > 0.5).astype(int)
    return {
        "loss": float(np.sum(losses) / max(n, 1)),
        "acc":  float(accuracy_score(l, preds)),
        "f1":   float(f1_score(l, preds, zero_division=0)),
        "auc":  float(roc_auc_score(l, p)) if len(set(l)) > 1 else float("nan"),
    }


# ── Train one variant ───────────────────────────────────────────────────────
def train_variant(name: str, scalar_keys: list[str], ckpt_out: Path,
                  img, txt, signals, labels, tr_idx, va_idx,
                  log_rows: list[dict],
                  lr_new_override: float | None = None) -> dict:
    # Per-variant seed reset: every variant starts from the same RNG state
    # so checkpoint comparison across variants is fair AND so Variant B
    # reproduces the Step-1 number independently of A/C/D having trained.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    lr_new_eff = LR_NEW if lr_new_override is None else lr_new_override
    scalar_dim = len(scalar_keys)
    print(f"\n{'='*72}\nVariant {name}: scalars = {scalar_keys}  (dim={scalar_dim})  lr_new={lr_new_eff:.0e}\n{'='*72}")

    scalar_stack = np.stack([signals[k] for k in scalar_keys], axis=1).astype(np.float32)

    n_fake_tr = int(labels[tr_idx].sum())
    n_real_tr = int((labels[tr_idx] == 0).sum())
    pos_w_val = n_real_tr / max(n_fake_tr, 1)
    print(f"  train fold: fake={n_fake_tr}  real={n_real_tr}  pos_weight={pos_w_val:.4f}")

    tr_ds = AITRFeatureDataset(img, txt, scalar_stack, labels, tr_idx)
    va_ds = AITRFeatureDataset(img, txt, scalar_stack, labels, va_idx)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, pin_memory=False)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=0, pin_memory=False)

    model = AITR(scalar_dim=scalar_dim).to(DEVICE)
    warm_start(model, AITR_INIT)  # skips scalar_proj since 9 != scalar_dim

    new_params, warm_params = [], []
    for nm, p in model.named_parameters():
        (new_params if nm.startswith("scalar_proj.") else warm_params).append(p)
    optim = torch.optim.AdamW(
        [{"params": new_params, "lr": lr_new_eff},
         {"params": warm_params, "lr": LR_WARM}],
        weight_decay=WEIGHT_DECAY,
    )

    def lr_lambda(epoch: int) -> float:
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / max(WARMUP_EPOCHS, 1)
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    pos_w = torch.tensor([pos_w_val], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    best_f1, best_epoch, stale = -1.0, -1, 0
    best_metrics = None
    t0 = time.time()
    for epoch in range(EPOCHS):
        tr = run_epoch(model, tr_dl, optim, loss_fn, train=True,
                       desc=f"V{name} ep{epoch+1}/{EPOCHS} [train]")
        with torch.no_grad():
            va = run_epoch(model, va_dl, optim, loss_fn, train=False,
                           desc=f"V{name} ep{epoch+1}/{EPOCHS} [val]  ")
        scheduler.step()
        print(f"  V{name} ep {epoch+1:02d} | "
              f"tr loss={tr['loss']:.4f} f1={tr['f1']:.4f} | "
              f"va loss={va['loss']:.4f} f1={va['f1']:.4f} acc={va['acc']:.4f} auc={va['auc']:.4f} | "
              f"lr_new={optim.param_groups[0]['lr']:.2e}")
        log_rows.append({
            "variant": name, "epoch": epoch + 1,
            "lr_new":  optim.param_groups[0]["lr"],
            "lr_warm": optim.param_groups[1]["lr"],
            "train_loss": tr["loss"], "val_loss": va["loss"],
            "val_acc": va["acc"], "val_f1": va["f1"], "val_auc": va["auc"],
        })

        if va["f1"] > best_f1:
            best_f1, best_epoch, stale = va["f1"], epoch + 1, 0
            best_metrics = va
            torch.save({
                "state_dict": model.state_dict(),
                "variant": name,
                "scalar_keys": scalar_keys,
                "scalar_dim": scalar_dim,
                "epoch": epoch + 1,
                "val_metrics": va,
                "pos_weight": pos_w_val,
            }, ckpt_out)
            print(f"    -> new best F1={best_f1:.4f}  saved {ckpt_out.name}")
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"    early stop (no improvement for {PATIENCE} epochs)")
                break

    print(f"  V{name} done in {time.time()-t0:.1f}s.  best F1={best_f1:.4f} @ epoch {best_epoch}")
    return {
        "variant": name, "scalar_keys": scalar_keys, "scalar_dim": scalar_dim,
        "best_epoch": best_epoch, "best_f1": best_f1, "best_metrics": best_metrics,
        "checkpoint": str(ckpt_out),
    }


# ── Held-out val sanity check (used to detect Variant-A degeneration) ──────
def load_heldout_val():
    sid = pd.read_csv(VAL_FEAT / "sample_ids.csv")
    img = torch.load(VAL_FEAT / "clip_img.pt", map_location="cpu", weights_only=False)
    txt = torch.load(VAL_FEAT / "clip_txt.pt", map_location="cpu", weights_only=False)
    clip_prob = np.load(VAL_FEAT / "clip_probs.npy").astype(np.float32)
    clip_sim  = np.load(VAL_FEAT / "clip_sims.npy").astype(np.float32)
    ateeq_df = pd.read_csv(ATEEQ_VAL)[["sample_id", "ateeq_score_ft"]]
    nli_df   = pd.read_csv(DEBERTA_VAL)[["sample_id", "deberta_score"]]
    m = (sid.merge(ateeq_df, on="sample_id", validate="one_to_one")
            .merge(nli_df,   on="sample_id", validate="one_to_one"))
    bad = m["ateeq_score_ft"] < 0
    if bad.any():
        med = float(m.loc[~bad, "ateeq_score_ft"].median())
        m.loc[bad, "ateeq_score_ft"] = med
    signals = {
        "clip_prob":     clip_prob,
        "clip_sim":      clip_sim,
        "ateeq_score":   m["ateeq_score_ft"].to_numpy(dtype=np.float32),
        "deberta_score": m["deberta_score"].to_numpy(dtype=np.float32),
    }
    return img, txt, signals, m["label"].to_numpy(dtype=np.int64)


@torch.no_grad()
def heldout_best_acc(ckpt_path: Path, val_img, val_txt, val_signals, val_labels) -> tuple[float, float]:
    """Returns (best_overall_acc, best_threshold) sweeping 0.10..0.70 step 0.05."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    keys = ckpt["scalar_keys"]
    scalar_dim = ckpt["scalar_dim"]
    scalar_stack = np.stack([val_signals[k] for k in keys], axis=1).astype(np.float32)
    model = AITR(scalar_dim=scalar_dim).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n = len(scalar_stack)
    probs = np.empty(n, dtype=np.float32)
    for s in range(0, n, 64):
        e = min(s + 64, n)
        i_n = F.normalize(val_img[s:e], dim=-1).to(DEVICE)
        t_n = F.normalize(val_txt[s:e], dim=-1).to(DEVICE)
        scl = torch.tensor(scalar_stack[s:e], dtype=torch.float32, device=DEVICE)
        probs[s:e] = torch.sigmoid(model(i_n, t_n, scl)).cpu().numpy()
    best_t, best_acc = 0.5, -1.0
    for t in np.arange(0.10, 0.7001, 0.05):
        acc = float(((probs > t).astype(int) == val_labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_acc, round(best_t, 2)


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Device: {DEVICE}")
    img, txt, signals, labels, _ = load_all_train_signals()
    n = len(labels)
    print(f"Loaded {n} train samples — fake={int(labels.sum())} real={int((labels==0).sum())}")

    # Shared stratified split — identical across variants
    idx_all = np.arange(n)
    tr_idx, va_idx = train_test_split(
        idx_all, test_size=0.2, random_state=SEED, stratify=labels,
    )
    print(f"Shared 80/20 split (seed={SEED}): train={len(tr_idx)}  val={len(va_idx)}")

    # Held-out val data (for A degeneration sanity check)
    val_img, val_txt, val_signals, val_labels = load_heldout_val()
    base_fake = float((val_labels == 1).mean())   # always-Fake floor (0.70)
    print(f"Held-out val: N={len(val_labels)}  always-Fake floor={base_fake*100:.2f}%")

    log_rows: list[dict] = []
    summary = []
    for name, cfg in VARIANTS.items():
        res = train_variant(
            name, cfg["scalars"], cfg["ckpt"],
            img, txt, signals, labels, tr_idx, va_idx, log_rows,
        )
        summary.append(res)

        # Sanity check on the held-out val
        acc_h, thr_h = heldout_best_acc(cfg["ckpt"], val_img, val_txt, val_signals, val_labels)
        print(f"  [sanity] {name} held-out best_acc={acc_h*100:.2f}% @ thr={thr_h}  "
              f"(always-Fake floor={base_fake*100:.2f}%)")
        res["heldout_acc"] = acc_h
        res["heldout_thr"] = thr_h

        # ── DISABLED (audit Prompt B, Part 4) ────────────────────────────────
        # The original code retrained Variant A whenever its best-threshold
        # accuracy on the EVALUATION val set equalled the always-Fake floor. That
        # keyed model selection to the eval-set sanity metric (val peeking /
        # model-selection leak). It is removed so the reported result is a single,
        # honest training run. If Variant A degenerates to always-Fake, report
        # that as the finding rather than retrying until it doesn't.
        #
        # (To keep an auto-retry honestly, it would have to gate on a SEPARATE
        #  held-out slice that is never used for evaluation — not val.)
        if name == "A" and abs(acc_h - base_fake) < 1e-3:
            print(f"  [NOTE] Variant A degenerated to always-Fake "
                  f"(acc={acc_h*100:.2f}% == floor {base_fake*100:.2f}%). "
                  f"Reporting as-is; auto-retrain disabled to avoid val peeking.")
            res["degenerate_always_fake"] = True

    pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
    print(f"\nLog -> {LOG_PATH}")

    print("\nSummary (internal-val + held-out sanity):")
    print(f"  {'var':<4} {'dim':<4} {'best_ep':<7} {'best_f1':<8} {'int_acc':<8} {'int_auc':<8} {'held_acc':<8} {'held_thr':<8}")
    for r in summary:
        bm = r["best_metrics"] or {}
        print(f"  {r['variant']:<4} {r['scalar_dim']:<4} {r['best_epoch']:<7} "
              f"{r['best_f1']:<8.4f} {bm.get('acc', float('nan')):<8.4f} "
              f"{bm.get('auc', float('nan')):<8.4f} "
              f"{r.get('heldout_acc', float('nan'))*100:<7.2f}% "
              f"{r.get('heldout_thr', float('nan')):<8}")


if __name__ == "__main__":
    main()
