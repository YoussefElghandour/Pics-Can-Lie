"""
2.5 — evaluate_ablation_mmfb.py
-------------------------------
Evaluate all 4 Step-2 AITR variants on the held-out MMFakeBench val (1000
samples). For each variant:
  - forward all 1000 samples (AITR-sigmoid probabilities)
  - sweep thresholds 0.05..0.95 step 0.05; pick best by overall accuracy
  - per-category accuracy at best threshold (original, mismatch,
    textual_veracity_distortion, visual_veracity_distortion)
  - AUC-ROC
  - confusion matrix overall + on textual_vd + on original specifically

Print THE thesis ablation table, plus analysis numbers:
  - Ateeq shortcut quantification: B.textual_vd - A.textual_vd
  - NLI legitimate contribution:    C.textual_vd - A.textual_vd
  - NLI on `original` FP recovery: C.original - A.original  and
                                    D.original - B.original
  - Best variant overall + best variant per-category

Save: eval_ablation_results.json
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
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_aitr_mmfb import AITR  # noqa: E402
from train_aitr_mmfb_v2 import VARIANTS  # noqa: E402

PROJECT_ROOT = Path(str(_cfg.ROOT))
OUT_ROOT     = PROJECT_ROOT / "mmfakebench_training"
VAL_FEAT     = OUT_ROOT / "val_features"
ATEEQ_VAL    = OUT_ROOT / "val_ateeq_scores_full.csv"
DEBERTA_VAL  = OUT_ROOT / "deberta_nli_val.csv"
OUT_JSON     = OUT_ROOT / "ablation_results_v2.json"

CATEGORIES = ["original", "mismatch", "textual_veracity_distortion", "visual_veracity_distortion"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_val():
    sid = pd.read_csv(VAL_FEAT / "sample_ids.csv")
    img = torch.load(VAL_FEAT / "clip_img.pt", map_location="cpu", weights_only=False)
    txt = torch.load(VAL_FEAT / "clip_txt.pt", map_location="cpu", weights_only=False)
    clip_prob = np.load(VAL_FEAT / "clip_probs.npy").astype(np.float32)
    clip_sim  = np.load(VAL_FEAT / "clip_sims.npy").astype(np.float32)
    ateeq = pd.read_csv(ATEEQ_VAL)[["sample_id", "ateeq_score_ft"]]
    nli   = pd.read_csv(DEBERTA_VAL)[["sample_id", "deberta_score"]]
    merged = (sid.merge(ateeq, on="sample_id", validate="one_to_one")
                 .merge(nli,   on="sample_id", validate="one_to_one"))
    bad = merged["ateeq_score_ft"] < 0
    if bad.any():
        med = float(merged.loc[~bad, "ateeq_score_ft"].median())
        merged.loc[bad, "ateeq_score_ft"] = med
    signals = {
        "clip_prob":     clip_prob,
        "clip_sim":      clip_sim,
        "ateeq_score":   merged["ateeq_score_ft"].to_numpy(dtype=np.float32),
        "deberta_score": merged["deberta_score"].to_numpy(dtype=np.float32),
    }
    labels = merged["label"].to_numpy(dtype=np.int64)
    cats   = merged["fake_cls"].to_numpy()
    return img, txt, signals, labels, cats


@torch.no_grad()
def forward_variant(ckpt_path: Path, img, txt, signals) -> tuple[np.ndarray, dict]:
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    keys = ckpt["scalar_keys"]
    scalar_dim = ckpt["scalar_dim"]
    scalar_stack = np.stack([signals[k] for k in keys], axis=1).astype(np.float32)

    model = AITR(scalar_dim=scalar_dim).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    n = len(scalar_stack)
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, 64):
        e = min(s + 64, n)
        i_n = F.normalize(img[s:e], dim=-1).to(DEVICE)
        t_n = F.normalize(txt[s:e], dim=-1).to(DEVICE)
        scl = torch.tensor(scalar_stack[s:e], dtype=torch.float32, device=DEVICE)
        out[s:e] = torch.sigmoid(model(i_n, t_n, scl)).cpu().numpy()
    meta = {
        "scalar_keys": keys,
        "scalar_dim":  scalar_dim,
        "epoch":       ckpt.get("epoch"),
        "pos_weight":  float(ckpt.get("pos_weight", float("nan"))),
    }
    return out, meta


def sweep_threshold(p: np.ndarray, y: np.ndarray):
    best_t, best_acc = 0.5, -1.0
    sweep = []
    for t in np.arange(0.10, 0.7001, 0.05):
        acc = float(((p > t).astype(int) == y).mean())
        sweep.append({"threshold": round(float(t), 2), "acc": acc})
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return round(best_t, 2), best_acc, sweep


def per_category_acc(preds: np.ndarray, labels: np.ndarray, cats: np.ndarray):
    res = OrderedDict()
    for c in CATEGORIES:
        m = cats == c
        if m.sum() == 0:
            res[c] = {"n": 0, "acc": float("nan")}
            continue
        res[c] = {"n": int(m.sum()), "acc": float((preds[m] == labels[m]).mean())}
    return res


def confusion_subset(preds: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> list:
    if mask.sum() == 0:
        return []
    return confusion_matrix(labels[mask], preds[mask], labels=[0, 1]).tolist()


def main() -> None:
    print(f"Device: {DEVICE}")
    img, txt, signals, labels, cats = load_val()
    n = len(labels)
    print(f"Val: N={n}  fake={int(labels.sum())}  real={int((labels==0).sum())}")
    print("  per-cat: " + "  ".join(f"{c}={int((cats==c).sum())}" for c in CATEGORIES))

    base_all_fake = float((labels == 1).mean())
    base_all_real = float((labels == 0).mean())
    print(f"  baselines: always-Fake={base_all_fake*100:.2f}%  always-Real={base_all_real*100:.2f}%")

    results = OrderedDict()
    for name in ["A", "B", "C", "D"]:
        ckpt = VARIANTS[name]["ckpt"]
        print(f"\n--- Variant {name} ({ckpt.name}) ---")
        probs, meta = forward_variant(ckpt, img, txt, signals)

        best_t, best_acc, sweep = sweep_threshold(probs, labels)
        preds = (probs > best_t).astype(int)
        auc = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")
        by_cat = per_category_acc(preds, labels, cats)
        cm_all = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
        cm_textual = confusion_subset(preds, labels, cats == "textual_veracity_distortion")
        cm_original = confusion_subset(preds, labels, cats == "original")
        acc_at_05 = float(((probs > 0.5).astype(int) == labels).mean())

        print(f"  scalars={meta['scalar_keys']}  best_t={best_t}  "
              f"overall={best_acc*100:.2f}%  AUC={auc:.4f}  acc@0.5={acc_at_05*100:.2f}%")
        for c, v in by_cat.items():
            print(f"    {c:<33} n={v['n']:<4} acc={v['acc']*100:.2f}%")

        results[name] = {
            "scalar_keys": meta["scalar_keys"],
            "scalar_dim":  meta["scalar_dim"],
            "best_epoch":  meta.get("epoch"),
            "pos_weight":  meta.get("pos_weight"),
            "best_threshold": best_t,
            "overall_acc":    best_acc,
            "acc_at_0.5":     acc_at_05,
            "auc_roc":        auc,
            "confusion_matrix_overall":    cm_all,
            "confusion_matrix_textual_vd": cm_textual,
            "confusion_matrix_original":   cm_original,
            "per_category":   {c: by_cat[c] for c in CATEGORIES},
            "threshold_sweep": sweep,
        }

    # ── THE thesis ablation table ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("ABLATION (val, 1000 samples) — accuracy at each variant's best threshold")
    print("=" * 100)
    header = f"{'CONFIGURATION':<22} | {'Overall':>7} | {'original':>8} | {'mismatch':>8} | {'textual_vd':>10} | {'visual_vd':>9} | {'AUC':>6} | {'thr':>4}"
    print(header)
    print("-" * len(header))
    names = {"A": "AITR (2 scalars)", "B": "+ Ateeq", "C": "+ NLI", "D": "+ Ateeq + NLI"}
    for k, label in names.items():
        r = results[k]
        pc = r["per_category"]
        print(f"{label:<22} | {r['overall_acc']*100:>6.2f}% | "
              f"{pc['original']['acc']*100:>7.2f}% | "
              f"{pc['mismatch']['acc']*100:>7.2f}% | "
              f"{pc['textual_veracity_distortion']['acc']*100:>9.2f}% | "
              f"{pc['visual_veracity_distortion']['acc']*100:>8.2f}% | "
              f"{r['auc_roc']:>6.3f} | {r['best_threshold']:>4.2f}")

    # ── Analysis numbers ─────────────────────────────────────────────────────
    A, B, C, D = (results[k] for k in "ABCD")
    def tvd(r): return r["per_category"]["textual_veracity_distortion"]["acc"]
    def org(r): return r["per_category"]["original"]["acc"]
    def msm(r): return r["per_category"]["mismatch"]["acc"]
    def vvd(r): return r["per_category"]["visual_veracity_distortion"]["acc"]

    ateeq_shortcut_tvd  = (tvd(B) - tvd(A)) * 100
    nli_legit_tvd       = (tvd(C) - tvd(A)) * 100
    nli_recover_orig_AC = (org(C) - org(A)) * 100
    nli_recover_orig_BD = (org(D) - org(B)) * 100
    additivity_pp       = (D["overall_acc"] - max(B["overall_acc"], C["overall_acc"])) * 100
    d_minus_b_pp        = (D["overall_acc"] - B["overall_acc"]) * 100

    print("\n" + "-" * 80)
    print("ANALYSIS")
    print("-" * 80)
    print(f"  [1] Ateeq shortcut quant.   (B.textual_vd - A.textual_vd): {ateeq_shortcut_tvd:+.2f} pp")
    print(f"  [2] NLI legitimate contrib  (C.textual_vd - A.textual_vd): {nli_legit_tvd:+.2f} pp")
    print(f"  [3] Additivity              (D.overall - max(B,C).overall): {additivity_pp:+.2f} pp")
    print(f"")
    print(f"  (aux) NLI FP recovery on `original` (C - A): {nli_recover_orig_AC:+.2f} pp")
    print(f"  (aux) NLI FP recovery on `original` (D - B): {nli_recover_orig_BD:+.2f} pp")

    # Verdict
    if d_minus_b_pp >= 2.0:
        verdict = "CLEAN WIN  (D beats B by >= 2 pp)"
    elif d_minus_b_pp >= -1.0:
        verdict = "BOUNDED POSITIVE  (-1 pp <= D - B < 2 pp)"
    else:
        verdict = "UNDERPERFORM  (D - B < -1 pp - investigate before concluding)"
    print(f"\n  D - B overall = {d_minus_b_pp:+.2f} pp  =>  {verdict}")

    # Step-1 regression: B should reproduce 72.6% +/- 0.5 pp
    reg_target = 0.726
    b_drift    = (B["overall_acc"] - reg_target) * 100
    reg_ok     = abs(B["overall_acc"] - reg_target) <= 0.005
    print(f"\n  [regression] B overall = {B['overall_acc']*100:.2f}%  "
          f"(target 72.60 +/- 0.50 pp, drift {b_drift:+.2f} pp) -> "
          f"{'OK' if reg_ok else 'DRIFT'}")

    best_overall = max(results, key=lambda k: results[k]["overall_acc"])
    print(f"\n  Best variant overall:           {best_overall}  "
          f"({results[best_overall]['overall_acc']*100:.2f}%)")
    for c in CATEGORIES:
        winner = max(results, key=lambda k: results[k]["per_category"][c]["acc"])
        print(f"  Best variant on {c:<33}: {winner}  "
              f"({results[winner]['per_category'][c]['acc']*100:.2f}%)")

    # ── Save ─────────────────────────────────────────────────────────────────
    payload = {
        "n_val": n,
        "baselines": {
            "always_predict_fake_acc": base_all_fake,
            "always_predict_real_acc": base_all_real,
        },
        "variants": results,
        "analysis": {
            "ateeq_shortcut_pp_tvd_BminusA":  ateeq_shortcut_tvd,
            "nli_contribution_pp_tvd_CminusA": nli_legit_tvd,
            "additivity_pp_D_minus_maxBC":    additivity_pp,
            "D_minus_B_overall_pp":           d_minus_b_pp,
            "verdict":                        verdict,
            "nli_original_delta_pp_CminusA":  nli_recover_orig_AC,
            "nli_original_delta_pp_DminusB":  nli_recover_orig_BD,
            "best_overall":                   best_overall,
            "B_regression_target":            reg_target,
            "B_overall":                      B["overall_acc"],
            "B_regression_drift_pp":          b_drift,
            "B_regression_ok":                bool(reg_ok),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
