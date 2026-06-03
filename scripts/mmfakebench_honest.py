"""
mmfakebench_honest.py  (Prompt B, Parts 1 & 2)

Part 1 — Retire the leaked "Wikipedia NLI veto": emit veto_honest_report.json with
         BOTH the oracle upper bound (uses ground-truth category, NOT deployable) and
         the real category-agnostic deployable number.
Part 2 — Honest MMFakeBench fusion comparison on the held-out val (1000), PRIMARY
         metric = AUC, accuracy only at a threshold frozen on a held-out split
         (here: a 20% calibration slice of the 10k train, never the eval set).
         Per-category accuracy + 1000-resample 95% bootstrap CIs for every model.
         Persists the best GBM to mmfakebench_gbm.joblib.

All from already-saved scores — no retraining of CLIP, no GPU needed.
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import config

ROOT = str(config.ROOT)
MT   = str(config.MMFB_TRAIN)            # experiments/mmfakebench_training
OVERNIGHT = str(config.EXPERIMENTS / "overnight_v2")
SEED = 42


# ───────────────────────── data ─────────────────────────
def load_split(feat_dir, ateeq_csv, deb_csv):
    sid = pd.read_csv(os.path.join(feat_dir, "sample_ids.csv"))
    cp = np.load(os.path.join(feat_dir, "clip_probs.npy")).astype(np.float32)
    cs = np.load(os.path.join(feat_dir, "clip_sims.npy")).astype(np.float32)
    ate = pd.read_csv(ateeq_csv)[["sample_id", "ateeq_score_ft"]]
    deb = pd.read_csv(deb_csv)[["sample_id", "deberta_score"]]
    m = sid.merge(ate, on="sample_id").merge(deb, on="sample_id")
    bad = m["ateeq_score_ft"] < 0
    if bad.any():
        m.loc[bad, "ateeq_score_ft"] = m.loc[~bad, "ateeq_score_ft"].median()
    X = np.stack([cp, cs, m["ateeq_score_ft"].values, m["deberta_score"].values], 1).astype(np.float32)
    return X, m["label"].values.astype(int), m["fake_cls"].values


def auc(y, p):
    order = np.argsort(p); r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def balanced_thr(y, p):
    grid = np.linspace(p.min(), p.max(), 501); best_t, best_b = 0.5, -1
    for t in grid:
        pred = (p >= t).astype(int)
        tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else 0
        tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else 0
        if 0.5 * (tpr + tnr) > best_b:
            best_b, best_t = 0.5 * (tpr + tnr), float(t)
    return best_t


def boot(y, p, thr, n=1000):
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); accs, aucs = [], []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2: continue
        accs.append(((p[b] >= thr).astype(int) == y[b]).mean()); aucs.append(auc(y[b], p[b]))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return q(accs), q(aucs)


def per_cat(y, p, thr, cats):
    out = {}
    for c in ["original", "mismatch", "textual_veracity_distortion", "visual_veracity_distortion"]:
        msk = cats == c
        if msk.any():
            out[c] = round(float(((p[msk] >= thr).astype(int) == y[msk]).mean()), 4)
    return out


def report(name, y, p, thr, cats):
    (acc_ci), (auc_ci) = boot(y, p, thr)
    return {"model": name, "AUC": round(auc(y, p), 4), "AUC_95ci": auc_ci,
            "accuracy": round(float(((p >= thr).astype(int) == y).mean()), 4),
            "accuracy_95ci": acc_ci, "threshold": round(thr, 4),
            "per_category": per_cat(y, p, thr, cats)}


# ───────────────── MMFB AITR (variant B, 3 scalars) ─────────────────
class AITRFusion(nn.Module):
    def __init__(self, num_scalars=3, embed_dim=768, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.scalar_proj = nn.Sequential(nn.Linear(num_scalars, embed_dim), nn.LayerNorm(embed_dim))
        self.type_embedding = nn.Embedding(5, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                         dim_feedforward=embed_dim * 2, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 256),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, img, txt, scl):
        B = img.size(0); te = self.type_embedding(torch.arange(5, device=img.device))
        tok = torch.cat([self.cls_token.expand(B, -1, -1),
                         (img + te[0]).unsqueeze(1), (txt + te[1]).unsqueeze(1),
                         (img * txt + te[2]).unsqueeze(1), (img - txt + te[3]).unsqueeze(1),
                         (self.scalar_proj(scl) + te[4]).unsqueeze(1)], 1).transpose(0, 1)
        return self.classifier(self.transformer(tok)[0]).squeeze(-1)


@torch.no_grad()
def aitr_probs(feat_dir, scl3):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    img = F.normalize(torch.load(os.path.join(feat_dir, "clip_img.pt"), map_location="cpu").float(), dim=-1)
    txt = F.normalize(torch.load(os.path.join(feat_dir, "clip_txt.pt"), map_location="cpu").float(), dim=-1)
    model = AITRFusion(num_scalars=3).to(dev).eval()
    ck = torch.load(os.path.join(MT, "aitr_mmfb_best.pt"), map_location=dev)
    model.load_state_dict(ck["state_dict"])
    out = np.empty(len(img), dtype=np.float32)
    for s in range(0, len(img), 256):
        e = min(s + 256, len(img))
        out[s:e] = torch.sigmoid(model(img[s:e].to(dev), txt[s:e].to(dev),
                                       torch.tensor(scl3[s:e], device=dev))).cpu().numpy()
    return out


# ───────────────────────── Part 1: veto ─────────────────────────
def part1_veto():
    ids = pd.read_csv(os.path.join(MT, "val_features", "sample_ids.csv"))
    deb = pd.read_csv(os.path.join(MT, "deberta_nli_val.csv"))
    labels = (ids["gt_answers"] == "Fake").astype(int).values
    fake_cls = ids["fake_cls"].values
    deberta = deb["deberta_score"].values
    t1 = json.load(open(os.path.join(OVERNIGHT, "task1_baseline.json")))
    base = (np.array(t1["all_probs"]) >= t1["best_threshold"]).astype(int)
    base_acc = float((base == labels).mean())
    thr = 0.05
    gated = base.copy()
    g = (fake_cls == "original") & (base == 1) & (deberta >= thr)
    gated[g] = 0
    agn = base.copy()
    a = (base == 1) & (deberta >= thr)
    agn[a] = 0
    fires = int(a.sum()); correct = int(((labels == 0) & a).sum())
    rep = {
        "baseline_overall_acc": round(base_acc, 4),
        "ORACLE_UPPER_BOUND": {
            "overall_acc": round(float((gated == labels).mean()), 4),
            "rule": "veto if fake_cls=='original' AND deberta>=0.05",
            "WARNING": "uses ground-truth MMFakeBench category (fake_cls) — NOT available at "
                       "inference, NOT deployable. Reports the ceiling only."},
        "DEPLOYABLE_category_agnostic": {
            "overall_acc": round(float((agn == labels).mean()), 4),
            "veto_precision": round(correct / fires, 4) if fires else None,
            "fires": fires, "correct_real": correct, "wrong_fake": fires - correct,
            "rule": "veto if deberta>=0.05 for any predicted-FAKE (no category gate)"},
        "note": "The veto reads deberta_score (article-NLI), not wiki_score, despite its name. "
                "Retained for the negative-finding discussion; never report as a headline."}
    json.dump(rep, open(str(config.RESULTS / "veto_honest_report.json"), "w"), indent=2)
    print("[Part 1] veto_honest_report.json written:")
    print(f"   ORACLE (not deployable) = {rep['ORACLE_UPPER_BOUND']['overall_acc']}")
    print(f"   DEPLOYABLE (agnostic)   = {rep['DEPLOYABLE_category_agnostic']['overall_acc']} "
          f"(precision {rep['DEPLOYABLE_category_agnostic']['veto_precision']})")
    return rep


# ───────────────────────── Part 2: fusion table ─────────────────────────
def part2_fusion():
    Xtr, ytr, _ = load_split(os.path.join(MT, "train_features"),
                             os.path.join(MT, "train_ateeq_scores.csv"),
                             os.path.join(MT, "deberta_nli_train.csv"))
    Xv, yv, catv = load_split(os.path.join(MT, "val_features"),
                              os.path.join(MT, "val_ateeq_scores_full.csv"),
                              os.path.join(MT, "deberta_nli_val.csv"))
    # 20% calibration slice of TRAIN to freeze thresholds (never the eval val).
    tr_i, cal_i = train_test_split(np.arange(len(ytr)), test_size=0.2,
                                   random_state=SEED, stratify=ytr)
    rows = []

    # Floor
    rows.append({"model": "always-predict-Fake (floor)", "AUC": None, "AUC_95ci": None,
                 "accuracy": round(float((yv == 1).mean()), 4), "accuracy_95ci": None,
                 "threshold": None, "per_category": per_cat(yv, np.ones_like(yv, float), 0.5, catv)})

    # AITR + Ateeq (variant B): freeze thr on train probs, eval on val.
    p_tr_aitr = aitr_probs(os.path.join(MT, "train_features"), Xtr[:, :3])
    p_v_aitr  = aitr_probs(os.path.join(MT, "val_features"),   Xv[:, :3])
    thr_a = balanced_thr(ytr, p_tr_aitr)
    rows.append(report("AITR + Ateeq (Variant B)", yv, p_v_aitr, thr_a, catv))

    # sklearn models
    best_gbm = None
    for name, cols, ctor in [
        ("GBM [clip_prob, clip_sim, ateeq]", [0, 1, 2], lambda: GradientBoostingClassifier(random_state=SEED)),
        ("GBM [clip_prob, clip_sim] (no Ateeq)", [0, 1], lambda: GradientBoostingClassifier(random_state=SEED)),
        ("LogReg [clip_prob, clip_sim, ateeq]", [0, 1, 2],
         lambda: LogisticRegression(max_iter=2000)),
    ]:
        is_lr = name.startswith("LogReg")
        sc = StandardScaler().fit(Xtr[tr_i][:, cols]) if is_lr else None
        Xt = sc.transform(Xtr[tr_i][:, cols]) if is_lr else Xtr[tr_i][:, cols]
        clf = ctor().fit(Xt, ytr[tr_i])
        def prob(M):
            return clf.predict_proba(sc.transform(M[:, cols]) if is_lr else M[:, cols])[:, 1]
        thr = balanced_thr(ytr[cal_i], prob(Xtr[cal_i]))
        rows.append(report(name, yv, prob(Xv), thr, catv))
        if name.startswith("GBM [clip_prob, clip_sim, ateeq]"):
            best_gbm = clf
            joblib.dump({"model": clf, "cols": cols,
                         "feature_order": ["clip_prob", "clip_sim", "ateeq_score_ft", "deberta_score"]},
                        str(config.GBM_PATH))

    out = {"n_val": len(yv), "primary_metric": "AUC",
           "threshold_protocol": "frozen on held-out train calibration slice (20%), never the eval val",
           "models": rows}
    json.dump(out, open(str(config.RESULTS / "mmfakebench_honest_comparison.json"), "w"), indent=2)

    print("\n[Part 2] mmfakebench_honest_comparison.json  (AUC primary)")
    print(f"{'model':<40}{'AUC':>8}{'AUC 95% CI':>20}{'acc':>7}  orig/mis/txt/vis")
    for r in rows:
        au = f"{r['AUC']:.4f}" if r["AUC"] is not None else "  -  "
        ci = f"[{r['AUC_95ci'][0]},{r['AUC_95ci'][1]}]" if r["AUC_95ci"] else " - "
        pc = r["per_category"]
        pcs = "/".join(f"{pc.get(c, float('nan')):.2f}" for c in
                       ["original", "mismatch", "textual_veracity_distortion", "visual_veracity_distortion"])
        print(f"{r['model']:<40}{au:>8}{ci:>20}{r['accuracy']:>7.4f}  {pcs}")
    print("   -> mmfakebench_gbm.joblib persisted (GBM on [clip_prob, clip_sim, ateeq])")


if __name__ == "__main__":
    part1_veto()
    part2_fusion()
