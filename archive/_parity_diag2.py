import json, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, clip
from PIL import Image
ROOT = r"D:\Pics Can Lie"; CF = os.path.join(ROOT, "clip_finetuned_v2", "val_features")
DEVICE = "cuda"
ids = pd.read_csv(os.path.join(CF, "val_sample_ids.csv")); ids["id"] = ids["id"].astype(str)
prob_pre = np.load(os.path.join(CF, "clip_finetuned_probs.npy")).astype(np.float32)
sim_pre  = np.load(os.path.join(CF, "clip_finetuned_sims.npy")).astype(np.float32)
print("val_sample_ids label balance:", dict(ids["label"].value_counts()))

class C(nn.Module):
    def __init__(s, m):
        super().__init__(); s.clip = m
        s.classifier = nn.Sequential(nn.Linear(768*2+1,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                                     nn.Linear(512,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128,1))
    def forward(s, im, t):
        i = F.normalize(s.clip.encode_image(im), dim=-1); x = F.normalize(s.clip.encode_text(t), dim=-1)
        c = (i*x).sum(-1, keepdim=True)
        return torch.sigmoid(s.classifier(torch.cat([i,x,c],-1))).squeeze(-1), i, x

meta = json.load(open(os.path.join(ROOT, r"dataset\data\NewsClipPings\metadata\val.json"), encoding="utf-8"))
IMG = os.path.join(ROOT, r"dataset\origin\origin")
def resolve(rel):
    rel = rel.replace("\\", "/")
    for p in ("visual_news/origin/", "origin/"):
        if rel.startswith(p):
            rel = rel[len(p):]; break
    return os.path.join(IMG, rel.replace("/", os.sep))

cb, pp = clip.load("ViT-L/14", device=DEVICE, jit=False)
clf = C(cb.float()).to(DEVICE)
clf.load_state_dict(torch.load(os.path.join(ROOT, "clip_finetuned_v2", "clip_classifier.pt"), map_location=DEVICE)["model_state"])
clf.eval()

# REAL rows only (label==0): id is its own image+caption, no falsified ambiguity.
real_pos = np.where(ids["label"].values == 0)[0]
rng = np.random.default_rng(0); pos = rng.choice(real_pos, min(120, len(real_pos)), replace=False)
imgs, toks, kp = [], [], []
for i in pos:
    sid = ids.iloc[int(i)]["id"]
    if sid not in meta: continue
    m = meta[sid]
    ip = resolve(m["image_path"])
    if not os.path.exists(ip): continue
    imgs.append(pp(Image.open(ip).convert("RGB")))
    toks.append(clip.tokenize([m["caption"]], truncate=True)[0]); kp.append(int(i))
print("resolved real rows:", len(kp), "/", len(pos))
with torch.no_grad():
    cp, i_f, t_f = clf(torch.stack(imgs).to(DEVICE), torch.stack(toks).to(DEVICE)); cs = (i_f*t_f).sum(-1)
cp = cp.cpu().numpy(); cs = cs.cpu().numpy(); kp = np.array(kp)
print("REAL rows  clip_prob fresh vs precomp r=%.4f  fresh %.4f / pre %.4f" % (np.corrcoef(cp, prob_pre[kp])[0,1], cp.mean(), prob_pre[kp].mean()))
print("REAL rows  clip_sim  fresh vs precomp r=%.4f  fresh %.4f / pre %.4f" % (np.corrcoef(cs, sim_pre[kp])[0,1], cs.mean(), sim_pre[kp].mean()))
