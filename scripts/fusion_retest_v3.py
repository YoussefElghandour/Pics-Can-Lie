"""
fusion_retest_v3.py  (Prompt B Part 2 / GPU pass — NLI re-test)

Swaps the corrected DeBERTa v3 scalar in for the broken v2 one and re-runs the
NewsCLIPpings AITR fusion (val protocol, train-only scaler, train-frozen
threshold). Reports:
  * DeBERTa v3 ALONE: REAL/FAKE mean entailment + AUC (is the NLI signal real?)
  * Fusion val AUC/accuracy with v3 vs the v2-based numbers (0.9477 / 0.880).

No retraining — reuses the existing AITR weights; the swap is inference-time, so
it measures whether a better NLI scalar would change the fused result.
"""
from __future__ import annotations
import json, os
import joblib, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import config

ROOT = str(config.ROOT)
CF = str(config.CLIP_VAL_FEATURES)
AITR_CKPT = str(config.AITR_CKPT)
V3_CSV = str(config.DEBERTA_V3_CSV)
V2_CSV = str(config.DEBERTA_V2_CSV)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
ORDER = ["clip_prob", "clip_sim", "deberta", "s2", "s3", "s4", "s5", "s6", "wiki"]


class AITR(nn.Module):
    def __init__(self, embed_dim=768, scalar_dim=9, num_heads=8, num_layers=2, dropout=0.3, hidden_dim=256):
        super().__init__()
        self.scalar_proj = nn.Sequential(nn.Linear(scalar_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.type_embedding = nn.Embedding(5, embed_dim)
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2,
                                         dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim)*0.02)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim,1))
    def forward(self, ie, te, s):
        B = ie.size(0)
        tok = torch.stack([ie, te, ie*te, ie-te, self.scalar_proj(s)], 1)
        tid = torch.arange(5, device=tok.device).unsqueeze(0).expand(B,-1)
        tok = tok + self.type_embedding(tid)
        out = self.transformer(torch.cat([self.cls_token.expand(B,-1,-1), tok], 1))
        return torch.sigmoid(self.classifier(out[:,0])).squeeze(-1)


def auc(y, p):
    order = np.argsort(p); r = np.empty(len(p)); r[order] = np.arange(1, len(p)+1)
    n1 = (y==1).sum(); n0 = (y==0).sum()
    return float((r[y==1].sum() - n1*(n1+1)/2)/(n1*n0))


def bal_thr(y, p):
    g = np.linspace(p.min(), p.max(), 501); bt, bb = 0.5, -1
    for t in g:
        pr = (p>=t).astype(int)
        tpr = (pr[y==1]==1).mean() if (y==1).any() else 0
        tnr = (pr[y==0]==0).mean() if (y==0).any() else 0
        if 0.5*(tpr+tnr) > bb: bb, bt = 0.5*(tpr+tnr), float(t)
    return bt


def boot(y, p, thr, n=1000):
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); a, b = [], []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2: continue
        a.append(((p[s]>=thr).astype(int)==y[s]).mean()); b.append(auc(y[s], p[s]))
    q = lambda z: [round(float(np.percentile(z,2.5)),4), round(float(np.percentile(z,97.5)),4)]
    return q(a), q(b)


def build(id_df, deberta_csv, deb_col):
    ids = id_df["id"].astype(str)
    cp = np.load(os.path.join(CF, "clip_finetuned_probs.npy")); cs = np.load(os.path.join(CF, "clip_finetuned_sims.npy"))
    d = pd.read_csv(deberta_csv); d["id"] = d["id"].astype(str)
    med = float(d[deb_col].median())              # impute unscored rows with valid-median
    dl = dict(zip(d["id"], d[deb_col]))
    deb = np.array([dl.get(i, med) for i in ids], dtype=np.float32)
    deb = np.where(np.isnan(deb), med, deb).astype(np.float32)
    ev = pd.read_csv(str(config.EVIDENCE_CSV)); ev["id"] = ev["id"].astype(str)
    el = {r["id"]: r for _, r in ev.iterrows()}
    S = {k: np.array([el.get(i, {}).get(k, 0.0) for i in ids], dtype=np.float32) for k in ["s2","s3","s4","s5","s6"]}
    wk = pd.read_csv(str(config.WIKI_NLI_CSV)); wk["id"] = wk["id"].astype(str)
    wl = {r["id"]: r for _, r in wk.iterrows()}; w1 = np.array([wl.get(i, {}).get("wiki_score", 0.33) for i in ids], dtype=np.float32)
    return np.column_stack([cp, cs, deb, S["s2"], S["s3"], S["s4"], S["s5"], S["s6"], w1]).astype(np.float32), deb


def run_fusion(raw, img, txt, labels, tr_idx, va_idx):
    scaler = StandardScaler().fit(raw[tr_idx]); scaled = scaler.transform(raw).astype(np.float32)
    model = AITR().to(DEVICE).eval()
    st = torch.load(AITR_CKPT, map_location=DEVICE); model.load_state_dict(st.get("state_dict", st))
    @torch.no_grad()
    def score(rows):
        out = np.empty(len(rows), np.float32)
        for s in range(0, len(rows), 256):
            e = min(s+256, len(rows)); b = rows[s:e]
            out[s:e] = model(img[b].to(DEVICE), txt[b].to(DEVICE), torch.tensor(scaled[b], device=DEVICE)).cpu().numpy()
        return out
    p_tr, p_va = score(tr_idx), score(va_idx)
    thr = bal_thr(labels[tr_idx], p_tr)
    acc = float(((p_va>=thr).astype(int)==labels[va_idx]).mean()); a = auc(labels[va_idx], p_va)
    (acc_ci), (auc_ci) = boot(labels[va_idx], p_va, thr)
    return {"AUC": round(a,4), "AUC_95ci": auc_ci, "accuracy": round(acc,4),
            "accuracy_95ci": acc_ci, "threshold": round(thr,4)}


def main():
    id_df = pd.read_csv(os.path.join(CF, "val_sample_ids.csv")); id_df["id"] = id_df["id"].astype(str)
    labels = id_df["label"].values.astype(int)
    img = F.normalize(torch.load(os.path.join(CF, "clip_img_features.pt")), dim=-1).float()
    txt = F.normalize(torch.load(os.path.join(CF, "clip_txt_features.pt")), dim=-1).float()
    idx = np.arange(len(labels)); tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=labels)

    # DeBERTa v3 ALONE
    raw_v3, deb_v3 = build(id_df, V3_CSV, "entailment_score")
    rm, fm = deb_v3[labels==0].mean(), deb_v3[labels==1].mean()
    # NLI "realness" = entailment; high => REAL. AUC of (1 - entailment) as fake-score:
    nli_auc = auc(labels, 1.0 - deb_v3)
    print(f"DeBERTa v3 ALONE: REAL mean={rm:.4f}  FAKE mean={fm:.4f}  gap={rm-fm:+.4f}")
    print(f"DeBERTa v3 ALONE AUC (as fake detector) = {nli_auc:.4f}")

    raw_v2, _ = build(id_df, V2_CSV, "entailment_score")
    res_v2 = run_fusion(raw_v2, img, txt, labels, tr_idx, va_idx)
    res_v3 = run_fusion(raw_v3, img, txt, labels, tr_idx, va_idx)
    print(f"\nFusion val (v2 deberta): {res_v2}")
    print(f"Fusion val (v3 deberta): {res_v3}")

    out = {"deberta_v3_alone": {"real_mean": float(rm), "fake_mean": float(fm),
                                "auc_as_fake_detector": round(nli_auc,4)},
           "fusion_val_v2": res_v2, "fusion_val_v3": res_v3,
           "delta_AUC": round(res_v3["AUC"]-res_v2["AUC"],4),
           "delta_acc": round(res_v3["accuracy"]-res_v2["accuracy"],4)}
    os.makedirs(str(config.RESULTS), exist_ok=True)
    json.dump(out, open(str(config.RESULTS / "deberta_v3_retest.json"), "w"), indent=2)
    print(f"\nDelta from v3 swap: AUC {out['delta_AUC']:+.4f}  acc {out['delta_acc']:+.4f}")
    print("Saved -> results/deberta_v3_retest.json")


if __name__ == "__main__":
    main()
