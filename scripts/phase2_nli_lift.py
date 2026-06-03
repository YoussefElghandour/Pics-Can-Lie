"""
phase2_nli_lift.py  (Roadmap Phase 2)

Precise number on whether DeBERTa v3 NLI adds anything over CLIP in the
NewsCLIPpings fusion. Cached features only (no CLIP retrain). Seconds to run.

Two simple fusions, scaler + threshold frozen on TRAIN only, eval on VAL:
    A) CLIP-only       [clip_prob, clip_sim]
    B) CLIP + v3 NLI   [clip_prob, clip_sim, deberta_v3]
with both LogisticRegression and GradientBoosting. AUC + 95% bootstrap CI,
accuracy delta. Also prints simple-fusion AUC vs the AITR AUC (0.9477) on the
same val split.

Saves results/fusion_nli_lift.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SEED = 42
AITR_VAL_AUC = 0.9477  # honest AITR val AUC (fix_newsclip_inference.py)


def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bal_thr(y, p):
    g = np.linspace(p.min(), p.max(), 501); bt, bb = 0.5, -1
    for t in g:
        pr = (p >= t).astype(int)
        tpr = (pr[y == 1] == 1).mean() if (y == 1).any() else 0
        tnr = (pr[y == 0] == 0).mean() if (y == 0).any() else 0
        if 0.5 * (tpr + tnr) > bb: bb, bt = 0.5 * (tpr + tnr), float(t)
    return bt


def boot(y, p, thr, n=1000):
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); a, b = [], []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2: continue
        a.append(((p[s] >= thr).astype(int) == y[s]).mean()); b.append(auc(y[s], p[s]))
    q = lambda z: [round(float(np.percentile(z, 2.5)), 4), round(float(np.percentile(z, 97.5)), 4)]
    return q(a), q(b)


def main():
    CF = config.CLIP_VAL_FEATURES
    ids = pd.read_csv(CF / "val_sample_ids.csv"); ids["id"] = ids["id"].astype(str)
    y = ids["label"].values.astype(int)
    cp = np.load(CF / "clip_finetuned_probs.npy").astype(np.float32)
    cs = np.load(CF / "clip_finetuned_sims.npy").astype(np.float32)
    d = pd.read_csv(config.DEBERTA_V3_CSV); d["id"] = d["id"].astype(str)
    med = float(d["entailment_score"].median())
    dl = dict(zip(d["id"], d["entailment_score"]))
    deb = np.array([dl.get(i, med) for i in ids["id"]], dtype=np.float32)
    deb = np.where(np.isnan(deb), med, deb).astype(np.float32)

    X = {"A_clip_only": np.column_stack([cp, cs]),
         "B_clip_plus_v3nli": np.column_stack([cp, cs, deb])}
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)

    models = {"LogisticRegression": lambda: LogisticRegression(max_iter=2000),
              "GradientBoosting":   lambda: GradientBoostingClassifier(random_state=SEED)}

    rows = []
    for feat_name, Xall in X.items():
        for mname, ctor in models.items():
            is_lr = mname == "LogisticRegression"
            sc = StandardScaler().fit(Xall[tr]) if is_lr else None
            Xt = sc.transform(Xall[tr]) if is_lr else Xall[tr]
            clf = ctor().fit(Xt, y[tr])
            pv = clf.predict_proba(sc.transform(Xall[va]) if is_lr else Xall[va])[:, 1]
            ptr = clf.predict_proba(sc.transform(Xall[tr]) if is_lr else Xall[tr])[:, 1]
            thr = bal_thr(y[tr], ptr)
            acc = float(((pv >= thr).astype(int) == y[va]).mean())
            (acc_ci), (auc_ci) = boot(y[va], pv, thr)
            rows.append({"features": feat_name, "model": mname, "AUC": round(auc(y[va], pv), 4),
                         "AUC_95ci": auc_ci, "accuracy": round(acc, 4), "accuracy_95ci": acc_ci,
                         "threshold": round(thr, 4)})

    # A vs B deltas per model
    deltas = {}
    for mname in models:
        a = next(r for r in rows if r["features"] == "A_clip_only" and r["model"] == mname)
        b = next(r for r in rows if r["features"] == "B_clip_plus_v3nli" and r["model"] == mname)
        deltas[mname] = {"delta_AUC": round(b["AUC"] - a["AUC"], 4),
                         "delta_acc": round(b["accuracy"] - a["accuracy"], 4),
                         "AUC_CIs_overlap": not (b["AUC_95ci"][0] > a["AUC_95ci"][1] or
                                                 a["AUC_95ci"][0] > b["AUC_95ci"][1])}

    best_simple_auc = max(r["AUC"] for r in rows)
    out = {"n_val": int(len(va)), "split": "80/20 stratified seed 42 (same as AITR)",
           "primary_metric": "AUC", "aitr_val_auc": AITR_VAL_AUC,
           "models": rows, "A_vs_B": deltas,
           "best_simple_fusion_AUC": best_simple_auc,
           "simple_vs_aitr": round(best_simple_auc - AITR_VAL_AUC, 4)}
    os.makedirs(config.RESULTS, exist_ok=True)
    json.dump(out, open(config.RESULTS / "fusion_nli_lift.json", "w"), indent=2)

    print(f"{'features':<20}{'model':<20}{'AUC':>8}{'AUC 95% CI':>20}{'acc':>8}")
    for r in rows:
        print(f"{r['features']:<20}{r['model']:<20}{r['AUC']:>8.4f}"
              f"{'['+str(r['AUC_95ci'][0])+','+str(r['AUC_95ci'][1])+']':>20}{r['accuracy']:>8.4f}")
    print("\nA (CLIP-only) vs B (CLIP + v3 NLI):")
    for m, dd in deltas.items():
        print(f"  {m:<20} dAUC={dd['delta_AUC']:+.4f}  dacc={dd['delta_acc']:+.4f}  "
              f"CIs overlap: {dd['AUC_CIs_overlap']}")
    print(f"\nBest simple-fusion AUC {best_simple_auc:.4f} vs AITR {AITR_VAL_AUC:.4f} "
          f"(delta {best_simple_auc - AITR_VAL_AUC:+.4f})")
    print("Saved -> results/fusion_nli_lift.json")


if __name__ == "__main__":
    main()
