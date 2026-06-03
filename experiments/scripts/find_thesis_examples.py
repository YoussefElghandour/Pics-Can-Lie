"""
Find 3 thesis-explanation samples from the val set:
  A — Wiki veto fired   (wiki_score >= 95th pctile of FAKE wiki, label=FAKE, AITR correct)
  B — Correct FAKE, CLIP dominant (clip_prob > 0.85, label=FAKE, low wiki, AITR correct)
  C — Correct REAL      (clip_prob < 0.15, label=REAL, AITR correct)
"""
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402


import os, json, gc, csv
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT       = str(_cfg.ROOT)
META       = rf"{ROOT}\dataset\data\NewsClipPings\metadata\val.json"
ANN        = rf"{ROOT}\dataset\data\NewsClipPings\merged_balanced\val.json"
FEAT_DIR   = rf"{ROOT}\clip_finetuned_v2\val_features"
CLIP_PROBS = rf"{FEAT_DIR}\clip_finetuned_probs.npy"
CLIP_SIMS  = rf"{FEAT_DIR}\clip_finetuned_sims.npy"
SAMPLE_IDS = rf"{FEAT_DIR}\val_sample_ids.csv"
IMG_FEATS  = rf"{FEAT_DIR}\clip_img_features.pt"
TXT_FEATS  = rf"{FEAT_DIR}\clip_txt_features.pt"
DEBERTA    = rf"{ROOT}\deberta_val_scores_v2.csv"
EVIDENCE   = rf"{ROOT}\evidence_clip_scores.csv"
WIKI       = rf"{ROOT}\wiki_nli_scores.csv"
AITR_CKPT  = rf"{ROOT}\fusion_aitr\aitr_weights.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Means for missing evidence/NLI scalars (same as test inference)
SCALAR_MEANS = {
    "deberta": 0.05, "s2": 0.731, "s3": 0.235, "s4": 0.620,
    "s5": 0.188, "s6": 0.573, "wiki": 0.05,
}
SOURCE_THRESHOLDS = {
    "bbc": 0.58, "washington_post": 0.57, "guardian": 0.54, "usa_today": 0.44,
}
DEFAULT_THRESHOLD = 0.54

# ── AITR architecture (matches fusion_aitr/aitr_weights.pt) ───────────────
class AITR(nn.Module):
    def __init__(self, embed_dim=768, scalar_dim=9, num_heads=8,
                 num_layers=2, dropout=0.3, hidden_dim=256):
        super().__init__()
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.type_embedding = nn.Embedding(5, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.classifier  = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(self, img, txt, scalar):
        B = img.size(0)
        scalar_emb = self.scalar_proj(scalar)
        tokens = torch.stack([img, txt, img*txt, img-txt, scalar_emb], dim=1)
        type_ids = torch.arange(5, device=tokens.device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.type_embedding(type_ids)
        cls = self.cls_token.expand(B, -1, -1)
        out = self.transformer(torch.cat([cls, tokens], dim=1))
        return torch.sigmoid(self.classifier(out[:, 0])).squeeze(-1)


# ── Load everything ───────────────────────────────────────────────────────
print("Loading metadata, annotations, scores ...")
with open(META, "r", encoding="utf-8") as f:
    meta = json.load(f)
with open(ANN, "r", encoding="utf-8") as f:
    ann_data = json.load(f)
label_map = {int(a["id"]): bool(a["falsified"]) for a in ann_data["annotations"]}
# Also keep the displayed image_id (differs when falsified)
imgid_map = {int(a["id"]): int(a["image_id"]) for a in ann_data["annotations"]}

deberta_df = pd.read_csv(DEBERTA, dtype={"id": str})
evid_df    = pd.read_csv(EVIDENCE, dtype={"id": str})
wiki_df    = pd.read_csv(WIKI, dtype={"id": str})

# val_sample_ids.csv → row order for the .npy files
sample_ids = []
csv_labels = []
with open(SAMPLE_IDS, "r", encoding="utf-8", newline="") as f:
    rdr = csv.reader(f)
    header = next(rdr)
    id_col    = header.index("id")
    label_col = header.index("label") if "label" in header else None
    for row in rdr:
        sample_ids.append(row[id_col])
        if label_col is not None:
            csv_labels.append(int(row[label_col]))
print(f"  val_sample_ids: {len(sample_ids)} (header={header})")

clip_probs = np.load(CLIP_PROBS)
clip_sims  = np.load(CLIP_SIMS)
print(f"  clip_probs shape={clip_probs.shape}, clip_sims shape={clip_sims.shape}")
assert len(sample_ids) == len(clip_probs) == len(clip_sims)

# Build per-id score table
def_score = lambda d, k: d.get(k, SCALAR_MEANS[k])
deberta_lookup = dict(zip(deberta_df["id"], deberta_df["entailment_score"].astype(float)))
wiki_lookup    = dict(zip(wiki_df["id"],    wiki_df["wiki_score"].astype(float)))
evid_lookup    = {row["id"]: {k: float(row[k]) for k in ["s2","s3","s4","s5","s6"]}
                  for _, row in evid_df.iterrows()}

# ── Build per-sample scalar matrix in the right order ─────────────────────
N = len(sample_ids)
scalars = np.zeros((N, 9), dtype=np.float32)
for i, sid in enumerate(sample_ids):
    scalars[i, 0] = clip_probs[i]
    scalars[i, 1] = clip_sims[i]
    scalars[i, 2] = deberta_lookup.get(sid, SCALAR_MEANS["deberta"])
    ev = evid_lookup.get(sid, {})
    scalars[i, 3] = ev.get("s2", SCALAR_MEANS["s2"])
    scalars[i, 4] = ev.get("s3", SCALAR_MEANS["s3"])
    scalars[i, 5] = ev.get("s4", SCALAR_MEANS["s4"])
    scalars[i, 6] = ev.get("s5", SCALAR_MEANS["s5"])
    scalars[i, 7] = ev.get("s6", SCALAR_MEANS["s6"])
    scalars[i, 8] = wiki_lookup.get(sid, SCALAR_MEANS["wiki"])

# ── Run AITR on cached features ──────────────────────────────────────────
print("Loading cached CLIP features ...")
img_feats = torch.load(IMG_FEATS, map_location="cpu").float()
txt_feats = torch.load(TXT_FEATS, map_location="cpu").float()
# AITR was trained on RAW (unnormalized) features — do NOT normalize here.
assert img_feats.shape[0] == N
print(f"  feature norms (raw): img~{img_feats.norm(dim=-1).mean():.2f}  "
      f"txt~{txt_feats.norm(dim=-1).mean():.2f}")

print("Loading AITR weights ...")
aitr = AITR()
state = torch.load(AITR_CKPT, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
aitr.load_state_dict(state)
del state; gc.collect()
aitr = aitr.to(DEVICE).eval()

print(f"Running AITR on {N} val samples ...")
all_probs = np.zeros(N, dtype=np.float32)
BS = 128
with torch.no_grad():
    for i in range(0, N, BS):
        j = min(i + BS, N)
        im = img_feats[i:j].to(DEVICE)
        tx = txt_feats[i:j].to(DEVICE)
        sc = torch.from_numpy(scalars[i:j]).to(DEVICE)
        all_probs[i:j] = aitr(im, tx, sc).cpu().numpy()
print(f"  AITR prob  min={all_probs.min():.4f}  max={all_probs.max():.4f}  mean={all_probs.mean():.4f}")

# ── Assemble the master table ────────────────────────────────────────────
rows = []
for i, sid in enumerate(sample_ids):
    sid_int  = int(sid)
    # Use the per-row label from val_sample_ids.csv (1=FAKE, 0=REAL).
    # Annotations are NOT safe to look up by id alone — the same id appears
    # in both REAL and FAKE pairs, so an id→label dict overwrites itself.
    if csv_labels:
        label = bool(csv_labels[i])
    else:
        label = label_map.get(sid_int)
    if label is None:
        continue
    cap_entry = meta.get(sid)
    if cap_entry is None:
        continue
    source = (cap_entry.get("source") or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    thr    = SOURCE_THRESHOLDS.get(source, DEFAULT_THRESHOLD)
    pred   = bool(all_probs[i] >= thr)
    rows.append({
        "id":         sid_int,
        "image_id":   imgid_map.get(sid_int, sid_int),
        "source":     source,
        "topic":      cap_entry.get("topic", ""),
        "caption":    cap_entry.get("caption", ""),
        "entities":   cap_entry.get("caption_entities_spacy", []),
        "clip_prob":  float(clip_probs[i]),
        "clip_sim":   float(clip_sims[i]),
        "deberta":    float(scalars[i, 2]),
        "s2": float(scalars[i, 3]), "s3": float(scalars[i, 4]),
        "s4": float(scalars[i, 5]), "s5": float(scalars[i, 6]),
        "s6": float(scalars[i, 7]),
        "wiki_score": float(scalars[i, 8]),
        "aitr_prob":  float(all_probs[i]),
        "threshold":  float(thr),
        "label":      bool(label),
        "pred":       pred,
        "correct":    pred == bool(label),
    })
df = pd.DataFrame(rows)
print(f"Joined table: {len(df)} rows  ({df['label'].sum()} FAKE, {(~df['label']).sum()} REAL)")
print(f"AITR overall val accuracy with source thresholds: {df['correct'].mean():.4f}")

# ── Wiki veto = 95th pctile of wiki_score among FAKE ─────────────────────
fake_wiki = df.loc[df["label"], "wiki_score"].values
WIKI_VETO = float(np.percentile(fake_wiki, 95))
print(f"\nWiki veto threshold (95th pctile of FAKE wiki_score) = {WIKI_VETO:.4f}")
print(f"  fake wiki_score: min={fake_wiki.min():.4f} med={np.median(fake_wiki):.4f} "
      f"p90={np.percentile(fake_wiki,90):.4f} p95={WIKI_VETO:.4f} max={fake_wiki.max():.4f}")

# ── Selection helpers ────────────────────────────────────────────────────
def show(title, sub):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    if len(sub) == 0:
        print("  (no matches)"); return
    for _, r in sub.iterrows():
        ents = [e[0] for e in (r["entities"] or [])][:6]
        print(f"\n  id={r['id']}  image_id={r['image_id']}  source={r['source']}  topic={r['topic']}")
        print(f"  caption : {r['caption'][:140]}")
        print(f"  entities: {ents}")
        print(f"  label={'FAKE' if r['label'] else 'REAL'}  pred={'FAKE' if r['pred'] else 'REAL'}  "
              f"aitr_prob={r['aitr_prob']:.4f}  thr={r['threshold']:.2f}")
        print(f"  clip_prob={r['clip_prob']:.4f}  clip_sim={r['clip_sim']:.4f}  "
              f"wiki={r['wiki_score']:.4f}  deberta={r['deberta']:.4f}")
        print(f"  evidence: s2={r['s2']:.3f}  s3={r['s3']:.3f}  s4={r['s4']:.3f}  "
              f"s5={r['s5']:.3f}  s6={r['s6']:.3f}")

# ── A — Wiki veto fired, FAKE, correct ───────────────────────────────────
A = (df[(df["wiki_score"] >= WIKI_VETO) & (df["label"]) & (df["correct"])]
       .sort_values("wiki_score", ascending=False).head(5))
show("Sample A — Wikipedia veto fired (wiki >= veto, label=FAKE, AITR correct)", A)

# ── B — CLIP dominant, FAKE, correct, wiki LOW ───────────────────────────
B = (df[(df["clip_prob"] > 0.85) & (df["label"]) & (df["correct"]) &
        (df["wiki_score"] < WIKI_VETO)]
       .sort_values("clip_prob", ascending=False).head(5))
show("Sample B — Correct FAKE, CLIP dominant (clip_prob>0.85, wiki below veto)", B)

# ── C — Correct REAL ─────────────────────────────────────────────────────
C = (df[(df["clip_prob"] < 0.15) & (~df["label"]) & (df["correct"])]
       .sort_values("clip_prob", ascending=True).head(5))
show("Sample C — Correct REAL (clip_prob<0.15, label=REAL, AITR correct)", C)

print("\nDone.")
