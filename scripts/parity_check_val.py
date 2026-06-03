"""
parity_check_val.py  — fresh-vs-precomputed equivalence gate (run BEFORE trusting
the test number). Val used PRECOMPUTED CLIP features; the test path computes them
FRESH. This recomputes a random 200-sample VAL slice fresh through the exact test
path and compares fused_prob to the precomputed-feature values.

PASS if Pearson r > 0.98 and mean abs diff is small.
"""
from __future__ import annotations
import json, os
import joblib, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import clip
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import config

ROOT = str(config.ROOT)
CF   = str(config.CLIP_VAL_FEATURES)
VAL_ANN  = str(config.VAL_ANN)
VAL_META = str(config.VAL_META)
IMAGES_ROOT = str(config.IMAGES_ROOT)
CLIP_CKPT = str(config.CLIP_CKPT)
AITR_CKPT = str(config.AITR_CKPT)
SCALER    = str(config.SCALER_PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 200
SEED = 42


class CLIPClassifier(nn.Module):
    def __init__(self, m):
        super().__init__(); self.clip = m
        self.classifier = nn.Sequential(
            nn.Linear(768*2+1,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128,1))
    def forward(self, images, ids):
        i = F.normalize(self.clip.encode_image(images), dim=-1)
        t = F.normalize(self.clip.encode_text(ids), dim=-1)
        c = (i*t).sum(-1, keepdim=True)
        return torch.sigmoid(self.classifier(torch.cat([i,t,c],-1))).squeeze(-1), i, t


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


def resolve(rel):
    rel = rel.replace("\\","/")
    for p in ("visual_news/origin/","origin/"):
        if rel.startswith(p): rel = rel[len(p):]; break
    return os.path.join(IMAGES_ROOT, rel.replace("/", os.sep))


def main():
    ids = pd.read_csv(os.path.join(CF, "val_sample_ids.csv")); ids["id"] = ids["id"].astype(str)
    img_pre = F.normalize(torch.load(os.path.join(CF,"clip_img_features.pt")), dim=-1).float()
    txt_pre = F.normalize(torch.load(os.path.join(CF,"clip_txt_features.pt")), dim=-1).float()
    prob_pre = np.load(os.path.join(CF,"clip_finetuned_probs.npy")).astype(np.float32)
    sim_pre  = np.load(os.path.join(CF,"clip_finetuned_sims.npy")).astype(np.float32)

    b = joblib.load(SCALER); scaler = b["scaler"]; tmeans = np.array(b["train_means"], dtype=np.float32)
    aitr = AITR().to(DEVICE).eval()
    st = torch.load(AITR_CKPT, map_location=DEVICE); aitr.load_state_dict(st.get("state_dict", st))

    clip_base, preprocess = clip.load("ViT-L/14", device=DEVICE, jit=False)
    clf = CLIPClassifier(clip_base.float()).to(DEVICE)
    clf.load_state_dict(torch.load(CLIP_CKPT, map_location=DEVICE)["model_state"]); clf.eval()

    # Key by (id, falsified): each article id has BOTH a real and a falsified
    # annotation; the val row's label selects which one (and thus which image).
    ann = {(str(a["id"]), bool(a["falsified"])): a
           for a in json.load(open(VAL_ANN, encoding="utf-8"))["annotations"]}
    meta = json.load(open(VAL_META, encoding="utf-8"))

    rng = np.random.default_rng(SEED)
    pos = rng.choice(len(ids), size=min(N, len(ids)), replace=False)

    def fused(img_emb, txt_emb, cprob, csim):
        raw = np.tile(tmeans, (len(cprob),1)).astype(np.float32)
        raw[:,0] = cprob; raw[:,1] = csim
        s = torch.tensor(scaler.transform(raw), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return aitr(img_emb.to(DEVICE), txt_emb.to(DEVICE), s).cpu().numpy()

    # precomputed-path fused
    f_pre = fused(img_pre[pos], txt_pre[pos], prob_pre[pos], sim_pre[pos])

    # fresh-path fused
    f_fresh = np.full(len(pos), np.nan, dtype=np.float32)
    imgs, toks, keep = [], [], []
    for k, i in enumerate(pos):
        row = ids.iloc[int(i)]; a = ann.get((str(row["id"]), bool(row["label"])))
        if a is None: continue
        img_key = str(a["image_id"]);
        if str(a["id"]) not in meta or img_key not in meta: continue
        ipath = resolve(meta[img_key]["image_path"])
        if not os.path.exists(ipath): continue
        imgs.append(preprocess(Image.open(ipath).convert("RGB")))
        toks.append(clip.tokenize([meta[str(a["id"])]["caption"]], truncate=True)[0])
        keep.append(k)
    print(f"Fresh-resolved {len(keep)}/{len(pos)} sampled val images")
    with torch.no_grad():
        ib = torch.stack(imgs).to(DEVICE); tb = torch.stack(toks).to(DEVICE)
        cprob, i_f, t_f = clf(ib, tb); csim = (i_f*t_f).sum(-1)
    f_fresh[keep] = fused(i_f.cpu(), t_f.cpu(), cprob.cpu().numpy(), csim.cpu().numpy())

    m = ~np.isnan(f_fresh)
    a, bb = f_pre[m], f_fresh[m]
    r = float(np.corrcoef(a, bb)[0,1]); mad = float(np.abs(a-bb).mean())
    print(f"\n=== PARITY (n={m.sum()}) ===")
    print(f"  Pearson r           = {r:.4f}")
    print(f"  mean abs diff       = {mad:.4f}")
    print(f"  precomp fused mean  = {a.mean():.4f}  fresh fused mean = {bb.mean():.4f}")
    verdict = "PASS" if (r > 0.98 and mad < 0.05) else "FAIL"
    print(f"  VERDICT: {verdict}  (gate: r>0.98 and small MAD)")
    os.makedirs(str(config.RESULTS), exist_ok=True)
    json.dump({"n": int(m.sum()), "pearson_r": r, "mean_abs_diff": mad,
               "precomp_mean": float(a.mean()), "fresh_mean": float(bb.mean()),
               "verdict": verdict}, open(str(config.RESULTS / "parity_check.json"),"w"), indent=2)


if __name__ == "__main__":
    main()
