import json, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, clip
from PIL import Image
ROOT = r"D:\Pics Can Lie"; CF = os.path.join(ROOT, "clip_finetuned_v2", "val_features")
DEVICE = "cuda"
ids = pd.read_csv(os.path.join(CF, "val_sample_ids.csv")); ids["id"] = ids["id"].astype(str)
prob_pre = np.load(os.path.join(CF, "clip_finetuned_probs.npy")).astype(np.float32)
sim_pre  = np.load(os.path.join(CF, "clip_finetuned_sims.npy")).astype(np.float32)

class C(nn.Module):
    def __init__(s, m):
        super().__init__(); s.clip = m
        s.classifier = nn.Sequential(nn.Linear(768*2+1,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                                     nn.Linear(512,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128,1))
    def forward(s, im, t):
        i = F.normalize(s.clip.encode_image(im), dim=-1); x = F.normalize(s.clip.encode_text(t), dim=-1)
        c = (i*x).sum(-1, keepdim=True)
        return torch.sigmoid(s.classifier(torch.cat([i,x,c],-1))).squeeze(-1), i, x

ann = {str(a["id"]): a for a in json.load(open(os.path.join(ROOT, r"dataset\data\NewsClipPings\merged_balanced\val.json"), encoding="utf-8"))["annotations"]}
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

rng = np.random.default_rng(42); pos = rng.choice(len(ids), 120, replace=False)
imgs, toks, kp, fals = [], [], [], []
for i in pos:
    a = ann.get(ids.iloc[int(i)]["id"])
    if a is None: continue
    ik = str(a["image_id"])
    if str(a["id"]) not in meta or ik not in meta: continue
    ip = resolve(meta[ik]["image_path"])
    if not os.path.exists(ip): continue
    imgs.append(pp(Image.open(ip).convert("RGB")))
    toks.append(clip.tokenize([meta[str(a["id"])]["caption"]], truncate=True)[0])
    kp.append(int(i)); fals.append(bool(a["falsified"]))

with torch.no_grad():
    cp, i_f, t_f = clf(torch.stack(imgs).to(DEVICE), torch.stack(toks).to(DEVICE)); cs = (i_f*t_f).sum(-1)
cp = cp.cpu().numpy(); cs = cs.cpu().numpy(); kp = np.array(kp); fals = np.array(fals)
print("clip_prob fresh vs precomp r=%.4f  (fresh mean %.4f vs precomp %.4f)" % (np.corrcoef(cp, prob_pre[kp])[0,1], cp.mean(), prob_pre[kp].mean()))
print("clip_sim  fresh vs precomp r=%.4f  (fresh mean %.4f vs precomp %.4f)" % (np.corrcoef(cs, sim_pre[kp])[0,1], cs.mean(), sim_pre[kp].mean()))
for f in [False, True]:
    m = fals == f
    print(" falsified=%s n=%d  clip_prob r=%.3f fresh %.3f/pre %.3f | clip_sim fresh %.3f/pre %.3f"
          % (f, m.sum(), np.corrcoef(cp[m], prob_pre[kp][m])[0,1], cp[m].mean(), prob_pre[kp][m].mean(), cs[m].mean(), sim_pre[kp][m].mean()))
