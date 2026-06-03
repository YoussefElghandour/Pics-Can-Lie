"""
1.5 — evaluate_aitr_mmfb.py
---------------------------
Evaluate the retrained AITR (mmfakebench_training/aitr_mmfb_best.pt) on the
held-out MMFakeBench val split (1000 samples — never seen during training).

For AITR-alone:
  - compute probabilities on all 1000 val samples
  - sweep thresholds 0.05..0.95 (step 0.05); pick best by overall accuracy
  - per-category accuracy at best threshold:
      original, mismatch, textual_veracity_distortion, visual_veracity_distortion
  - AUC-ROC, confusion matrix
  - save mmfakebench_training/eval_results_aitr_only.json

Then AITR + Ateeq fusion (OR rule):
  FAKE if  ateeq_score_ft > 0.5  OR  aitr_prob > best_threshold
  - same per-category breakdown
  - save mmfakebench_training/eval_results_aitr_plus_ateeq.json

Finally print a comparison table vs. the published zero-shot AITR numbers.
"""


from __future__ import annotations
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm

from train_aitr_mmfb import AITR  # re-use the exact architecture

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(str(_cfg.ROOT))
OUT_ROOT     = PROJECT_ROOT / "mmfakebench_training"
VAL_FEAT     = OUT_ROOT / "val_features"
ATEEQ_VAL    = OUT_ROOT / "val_ateeq_scores_full.csv"
BEST_PATH    = OUT_ROOT / "aitr_mmfb_best.pt"

OUT_AITR     = OUT_ROOT / "eval_results_aitr_only.json"
OUT_FUSION   = OUT_ROOT / "eval_results_aitr_plus_ateeq.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Zero-shot numbers (your old NewsCLIPpings-trained AITR on MMFakeBench)
ZERO_SHOT = {
    "Overall":                     58.5,
    "original":                    65.0,
    "mismatch":                    59.0,
    "textual_veracity_distortion": 40.0,
    "visual_veracity_distortion":  93.0,
}
CATEGORIES = ["original", "mismatch", "textual_veracity_distortion", "visual_veracity_distortion"]


def load_val():
    sid = pd.read_csv(VAL_FEAT / "sample_ids.csv")
    img = torch.load(VAL_FEAT / "clip_img.pt", map_location="cpu", weights_only=False)
    txt = torch.load(VAL_FEAT / "clip_txt.pt", map_location="cpu", weights_only=False)
    probs = np.load(VAL_FEAT / "clip_probs.npy")
    sims  = np.load(VAL_FEAT / "clip_sims.npy")

    ateeq = pd.read_csv(ATEEQ_VAL)
    merged = sid.merge(
        ateeq[["sample_id", "ateeq_score_ft"]],
        on="sample_id", how="left", validate="one_to_one",
    )
    if merged["ateeq_score_ft"].isna().any():
        raise RuntimeError("val rows missing ateeq_score_ft — re-run 1.3")
    bad = merged["ateeq_score_ft"] < 0
    if bad.any():
        med = float(merged.loc[~bad, "ateeq_score_ft"].median())
        print(f"[val] {int(bad.sum())} ateeq sentinel(-1) -> median {med:.3f}")
        merged.loc[bad, "ateeq_score_ft"] = med
    return merged, img, txt, probs.astype(np.float32), sims.astype(np.float32), \
           merged["ateeq_score_ft"].to_numpy(dtype=np.float32), \
           merged["label"].to_numpy(dtype=np.int64), \
           merged["fake_cls"].to_numpy()


@torch.no_grad()
def aitr_probs(model, img, txt, probs, sims, ateeq, batch_size: int = 64) -> np.ndarray:
    n = len(img)
    out = np.empty(n, dtype=np.float32)
    model.eval()
    for s in tqdm(range(0, n, batch_size), desc="AITR forward"):
        e = min(s + batch_size, n)
        i_n = F.normalize(img[s:e], dim=-1).to(DEVICE)
        t_n = F.normalize(txt[s:e], dim=-1).to(DEVICE)
        scl = torch.tensor(
            np.stack([probs[s:e], sims[s:e], ateeq[s:e]], axis=1),
            dtype=torch.float32, device=DEVICE,
        )
        logits = model(i_n, t_n, scl)
        out[s:e] = torch.sigmoid(logits).cpu().numpy()
    return out


def per_category_acc(preds: np.ndarray, labels: np.ndarray, cats: np.ndarray) -> "OrderedDict[str, dict]":
    res: "OrderedDict[str, dict]" = OrderedDict()
    for c in CATEGORIES:
        mask = cats == c
        if mask.sum() == 0:
            res[c] = {"n": 0, "acc": float("nan")}
            continue
        acc = float((preds[mask] == labels[mask]).mean())
        res[c] = {"n": int(mask.sum()), "acc": acc}
    return res


def sweep_threshold(p: np.ndarray, y: np.ndarray):
    best_t, best_acc = 0.5, -1.0
    sweep = []
    for t in np.arange(0.05, 0.96, 0.05):
        acc = float(((p > t).astype(int) == y).mean())
        sweep.append({"threshold": round(float(t), 2), "acc": acc})
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return round(best_t, 2), best_acc, sweep


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Loading val features ...")
    merged, img, txt, clip_probs, clip_sims, ateeq, labels, cats = load_val()
    print(f"  N={len(labels)}  fake={labels.sum()}  real={(labels==0).sum()}")
    print(f"  per-category:  " + "  ".join(f"{c}={int((cats==c).sum())}" for c in CATEGORIES))

    ckpt = torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)
    print(f"Loading AITR from {BEST_PATH.name}  (epoch={ckpt.get('epoch','?')})")
    model = AITR(scalar_dim=3).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    # ── Baselines for honest interpretation ──────────────────────────────────
    pos_weight = float(ckpt.get("pos_weight", float("nan")))
    base_all_fake = float((labels == 1).mean())   # accuracy of predicting Fake everywhere
    base_all_real = float((labels == 0).mean())   # accuracy of predicting Real everywhere
    print(f"\n[baselines]")
    print(f"  pos_weight from train fold     = {pos_weight:.4f}")
    print(f"  always-predict-Fake (val acc)  = {base_all_fake*100:.2f}%   <-- floor model must beat")
    print(f"  always-predict-Real (val acc)  = {base_all_real*100:.2f}%")

    p_aitr = aitr_probs(model, img, txt, clip_probs, clip_sims, ateeq)
    acc_at_05 = float(((p_aitr > 0.5).astype(int) == labels).mean())
    print(f"  AITR acc at threshold=0.50     = {acc_at_05*100:.2f}%")

    # ── AITR-only: sweep threshold by overall accuracy ───────────────────────
    best_t, best_acc, sweep = sweep_threshold(p_aitr, labels)
    preds_aitr = (p_aitr > best_t).astype(int)
    auc = float(roc_auc_score(labels, p_aitr)) if len(set(labels)) > 1 else float("nan")
    cm = confusion_matrix(labels, preds_aitr).tolist()
    by_cat_aitr = per_category_acc(preds_aitr, labels, cats)

    res_aitr = {
        "n": int(len(labels)),
        "best_threshold": best_t,
        "overall_acc": best_acc,
        "acc_at_0.5": acc_at_05,
        "auc_roc": auc,
        "confusion_matrix": cm,
        "per_category": by_cat_aitr,
        "threshold_sweep": sweep,
        "baselines": {
            "pos_weight_train": pos_weight,
            "always_predict_fake_acc": base_all_fake,
            "always_predict_real_acc": base_all_real,
        },
        "pos_label": "Fake",
        "checkpoint": str(BEST_PATH),
    }
    OUT_AITR.write_text(json.dumps(res_aitr, indent=2))
    print(f"\nAITR-only  best_t={best_t}  acc={best_acc*100:.2f}%  AUC={auc:.4f}")
    for c, v in by_cat_aitr.items():
        print(f"  {c:<32} n={v['n']:<4}  acc={v['acc']*100:.2f}%")

    # ── Fusion (OR rule): FAKE if ateeq>0.5 OR aitr>best_t ───────────────────
    preds_fusion = (((ateeq > 0.5) | (p_aitr > best_t))).astype(int)
    acc_fusion = float((preds_fusion == labels).mean())
    cm_fusion = confusion_matrix(labels, preds_fusion).tolist()
    by_cat_fusion = per_category_acc(preds_fusion, labels, cats)

    res_fusion = {
        "n": int(len(labels)),
        "rule": "FAKE if (ateeq_score_ft > 0.5) OR (aitr_prob > best_threshold)",
        "best_threshold_aitr": best_t,
        "overall_acc": acc_fusion,
        "confusion_matrix": cm_fusion,
        "per_category": by_cat_fusion,
    }
    OUT_FUSION.write_text(json.dumps(res_fusion, indent=2))
    print(f"\nAITR + Ateeq (OR)  acc={acc_fusion*100:.2f}%")
    for c, v in by_cat_fusion.items():
        print(f"  {c:<32} n={v['n']:<4}  acc={v['acc']*100:.2f}%")

    # ── Baselines summary block (honest interpretation aids) ────────────────
    print("\n" + "-" * 78)
    print("BASELINES & THRESHOLD-TUNING CHECK")
    print("-" * 78)
    print(f"  always-predict-Fake on val            = {base_all_fake*100:6.2f}%  (floor)")
    print(f"  always-predict-Real on val            = {base_all_real*100:6.2f}%")
    print(f"  pos_weight from train fold (n_real/n_fake) = {pos_weight:.4f}")
    print(f"  AITR @ thr=0.50                       = {acc_at_05*100:6.2f}%")
    print(f"  AITR @ best thr={best_t:<5}                = {best_acc*100:6.2f}%  "
          f"({'+' if best_acc-acc_at_05>=0 else ''}{(best_acc-acc_at_05)*100:.2f}% from threshold tuning)")
    print(f"  AITR vs always-Fake                   = "
          f"{('+' if best_acc-base_all_fake>=0 else '')}{(best_acc-base_all_fake)*100:.2f}%  "
          f"(positive => model is actually learning, not just exploiting imbalance)")

    # ── Comparison table ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'':<32} | {'Zero-shot (old)':>16} | {'Retrained (new)':>16} | {'Change':>8}")
    print("-" * 78)
    rows = [("Overall", best_acc * 100)] + [
        (c, by_cat_aitr[c]["acc"] * 100) for c in CATEGORIES
    ]
    for name, new_val in rows:
        old = ZERO_SHOT.get(name, float("nan"))
        delta = new_val - old
        sign = "+" if delta >= 0 else ""
        print(f"{name:<32} | {old:>15.1f}% | {new_val:>15.1f}% | {sign}{delta:>6.1f}")
    print("=" * 78)
    print("\nAITR + Ateeq (OR) — for reference:")
    print(f"{'Overall':<32} | {'':>15}  | {acc_fusion*100:>15.1f}% | "
          f"{('+' if acc_fusion*100 - ZERO_SHOT['Overall']>=0 else '')}"
          f"{acc_fusion*100 - ZERO_SHOT['Overall']:>6.1f}")
    for c in CATEGORIES:
        v = by_cat_fusion[c]["acc"] * 100
        old = ZERO_SHOT.get(c, float("nan"))
        d = v - old
        s = "+" if d >= 0 else ""
        print(f"{c:<32} | {old:>15.1f}% | {v:>15.1f}% | {s}{d:>6.1f}")

    print(f"\nSaved: {OUT_AITR.name}, {OUT_FUSION.name}")


if __name__ == "__main__":
    main()
