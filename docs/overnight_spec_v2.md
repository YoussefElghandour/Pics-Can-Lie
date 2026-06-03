# Overnight Spec v2 — MMFakeBench Deep Analysis
> Working directory: `D:\Pics Can Lie\`
> Output directory: `D:\Pics Can Lie\overnight_v2\` (create if not exists)
> Run tasks in order. If a task fails, write error to DONE_v2.md and continue.
> Use `.\venv\Scripts\python.exe` for all execution.
> Save every script to `D:\Pics Can Lie\overnight_v2\` before running.
> DO NOT retrain anything. DO NOT modify existing files. Inference and analysis only.
> When all done: print "ALL TASKS COMPLETE — check DONE_v2.md"

---

## Verified Configuration (do not deviate)

### MMFakeBench AITR Architecture (scalar_dim=3, verified)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AITRFusion(nn.Module):
    def __init__(self, embed_dim=768, num_scalars=3, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.scalar_proj = nn.Sequential(
            nn.Linear(num_scalars, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.type_embedding = nn.Embedding(5, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, img_emb, txt_emb, scalars):
        B = img_emb.size(0)
        img_times_txt = img_emb * txt_emb
        img_minus_txt = img_emb - txt_emb
        scalar_token  = self.scalar_proj(scalars)
        type_ids  = torch.arange(5, device=img_emb.device)
        type_embs = self.type_embedding(type_ids)
        tokens = torch.cat([
            self.cls_token.expand(B, -1, -1),
            (img_emb       + type_embs[0]).unsqueeze(1),
            (txt_emb       + type_embs[1]).unsqueeze(1),
            (img_times_txt + type_embs[2]).unsqueeze(1),
            (img_minus_txt + type_embs[3]).unsqueeze(1),
            (scalar_token  + type_embs[4]).unsqueeze(1),
        ], dim=1).transpose(0, 1)   # (6, B, 768)
        out    = self.transformer(tokens)
        cls_out = out[0]            # (B, 768)
        return self.classifier(cls_out).squeeze(-1)


def load_aitr_mmfb(path, device):
    model = AITRFusion(num_scalars=3)
    ckpt  = torch.load(path, map_location=device)
    state = ckpt['state_dict']
    res   = model.load_state_dict(state, strict=True)
    print(f"Loaded MMFakeBench AITR. epoch={ckpt['epoch']}, "
          f"saved_acc={ckpt['val_metrics']['acc']:.4f}, "
          f"missing={res.missing_keys}, unexpected={res.unexpected_keys}")
    return model.to(device).eval()
```

### File Paths (verified)
```python
MMFB_AITR_PATH  = r'D:\Pics Can Lie\mmfakebench_training\aitr_mmfb_best.pt'
VAL_IMG_PT      = r'D:\Pics Can Lie\mmfakebench_training\val_features\clip_img.pt'
VAL_TXT_PT      = r'D:\Pics Can Lie\mmfakebench_training\val_features\clip_txt.pt'
VAL_PROBS_NPY   = r'D:\Pics Can Lie\mmfakebench_training\val_features\clip_probs.npy'
VAL_SIMS_NPY    = r'D:\Pics Can Lie\mmfakebench_training\val_features\clip_sims.npy'
VAL_IDS_CSV     = r'D:\Pics Can Lie\mmfakebench_training\val_features\sample_ids.csv'
ATEEQ_CSV       = r'D:\Pics Can Lie\mmfakebench_training\val_ateeq_scores_full.csv'
DEBERTA_CSV     = r'D:\Pics Can Lie\mmfakebench_training\deberta_nli_val.csv'
MMFB_VAL_JSON   = r'E:\Pics Can Lie\dataset\MMFakeBench\MMFakeBench_val.json'
OUT_DIR         = r'D:\Pics Can Lie\overnight_v2'
```

### Data Loading Helper (use in every task)
```python
import torch, numpy as np, pandas as pd, json, os
import torch.nn.functional as F

os.makedirs(OUT_DIR, exist_ok=True)

def load_mmfb_val():
    """Load and align all MMFakeBench val features. Returns dict."""
    # Load features
    img_raw = torch.load(VAL_IMG_PT, map_location='cpu').float()
    txt_raw = torch.load(VAL_TXT_PT, map_location='cpu').float()
    img     = F.normalize(img_raw, dim=-1)
    txt     = F.normalize(txt_raw, dim=-1)
    probs   = np.load(VAL_PROBS_NPY)   # (1000,) fine-tuned CLIP probs
    sims    = np.load(VAL_SIMS_NPY)    # (1000,) cosine similarities
    ids_df  = pd.read_csv(VAL_IDS_CSV) # has idx(0-999), sample_id, fake_cls, gt_answers, label

    # Load ateeq — align by position (both are 0-999 in order)
    ateeq_df  = pd.read_csv(ATEEQ_CSV)
    deberta_df = pd.read_csv(DEBERTA_CSV)

    # Verify alignment
    assert len(img) == len(ids_df) == len(ateeq_df) == len(deberta_df) == 1000, "Length mismatch!"

    # Labels: gt_answers uses 'Fake'/'True' (NOT 'False')
    labels    = (ids_df['gt_answers'] == 'Fake').astype(int).values
    fake_cls  = ids_df['fake_cls'].values
    categories = ['original', 'mismatch', 'textual_veracity_distortion', 'visual_veracity_distortion']

    # Scalars: [clip_prob, clip_sim, ateeq_score] — scalar_dim=3
    ateeq_scores   = ateeq_df['ateeq_score_ft'].values
    deberta_scores = deberta_df['deberta_score'].values

    scalars = torch.tensor(
        np.stack([probs, sims, ateeq_scores], axis=1), dtype=torch.float32
    )  # (1000, 3)

    return {
        'img': img, 'txt': txt,
        'probs': probs, 'sims': sims,
        'ateeq': ateeq_scores, 'deberta': deberta_scores,
        'scalars': scalars,
        'labels': labels,
        'fake_cls': fake_cls,
        'categories': categories,
        'ids_df': ids_df,
    }
```

---

## Task 1 — Baseline Reproduction + Threshold Investigation

**Goal:** Reproduce 72.6% and investigate why saved val_metrics shows 74.05%.

Save as `overnight_v2/task1_baseline.py` and run.

```python
import torch, numpy as np, pandas as pd, json, os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# paste AITRFusion class, load_aitr_mmfb, file paths, and load_mmfb_val here

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

data  = load_mmfb_val()
model = load_aitr_mmfb(MMFB_AITR_PATH, device)

img     = data['img'].to(device)
txt     = data['txt'].to(device)
scalars = data['scalars'].to(device)
labels  = data['labels']
fake_cls = data['fake_cls']
categories = data['categories']

# Run inference
all_probs = []
with torch.no_grad():
    for i in range(0, 1000, 64):
        logits = model(img[i:i+64], txt[i:i+64], scalars[i:i+64])
        all_probs.append(torch.sigmoid(logits).cpu())
all_probs = torch.cat(all_probs).numpy()

auc = roc_auc_score(labels, all_probs)
print(f"\nAUC-ROC: {auc:.4f}")

# Full threshold sweep 0.05 to 0.95
thresholds = np.arange(0.05, 0.96, 0.05)
sweep = []
for thr in thresholds:
    preds = (all_probs >= thr).astype(int)
    acc   = (preds == labels).mean()
    per_cat = {}
    for cat in categories:
        mask = fake_cls == cat
        per_cat[cat] = round(float((preds[mask] == labels[mask]).mean()), 4)
    sweep.append({
        'threshold': round(float(thr), 2),
        'overall':   round(float(acc), 4),
        'per_cat':   per_cat
    })
    print(f"thr={thr:.2f}: overall={acc:.4f} | "
          + " | ".join(f"{c[:4]}={per_cat[c]:.4f}" for c in categories))

best = max(sweep, key=lambda x: x['overall'])
print(f"\nBest overall: thr={best['threshold']}, acc={best['overall']:.4f}")
print(f"At thr=0.30 (reported): {next(s for s in sweep if s['threshold']==0.30)['overall']:.4f}")

# Also check: what threshold maximizes each category
for cat in categories:
    best_cat = max(sweep, key=lambda x: x['per_cat'][cat])
    print(f"Best thr for {cat}: {best_cat['threshold']} → {best_cat['per_cat'][cat]:.4f}")

prec, rec, f1, _ = precision_recall_fscore_support(
    labels, (all_probs >= best['threshold']).astype(int), average='binary')

output = {
    'auc_roc': round(float(auc), 4),
    'best_threshold': best['threshold'],
    'best_overall': best['overall'],
    'best_per_cat': best['per_cat'],
    'at_030': next(s for s in sweep if s['threshold'] == 0.30),
    'precision': round(float(prec), 4),
    'recall':    round(float(rec), 4),
    'f1':        round(float(f1), 4),
    'full_sweep': sweep,
    'all_probs': all_probs.tolist(),
}
with open(os.path.join(OUT_DIR, 'task1_baseline.json'), 'w') as f:
    json.dump(output, f, indent=2)
print("Task 1 DONE")
```

---

## Task 2 — Per-Category Threshold Optimization

**Goal:** Give each category its own optimal threshold. This mirrors the source-aware thresholds on NewsCLIPpings (+0.20% there). Expected gain: +1-3% overall from recovering `original`.

Save as `overnight_v2/task2_per_category_thresholds.py` and run.

```python
import torch, numpy as np, pandas as pd, json, os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from itertools import product

# paste AITRFusion, load_aitr_mmfb, paths, load_mmfb_val here

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data   = load_mmfb_val()

# Load saved probs from Task 1 (don't re-run inference)
with open(os.path.join(OUT_DIR, 'task1_baseline.json')) as f:
    t1 = json.load(f)
all_probs = np.array(t1['all_probs'])
labels    = data['labels']
fake_cls  = data['fake_cls']
categories = data['categories']

# ── Strategy 1: Independent per-category threshold ──
# For each category, find the threshold that maximizes that category's accuracy
# Then combine: each sample uses its category's threshold
print("=== Strategy 1: Independent Per-Category Thresholds ===")
cat_thresholds = {}
for cat in categories:
    mask     = fake_cls == cat
    cat_labs = labels[mask]
    cat_prob = all_probs[mask]
    best_acc, best_thr = 0, 0.30
    for thr in np.arange(0.05, 0.96, 0.05):
        acc = ((cat_prob >= thr).astype(int) == cat_labs).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    cat_thresholds[cat] = {'threshold': round(float(best_thr), 2),
                           'accuracy':  round(float(best_acc), 4)}
    print(f"  {cat}: thr={best_thr:.2f} → {best_acc:.4f}")

# Apply per-category thresholds
preds_cat = np.zeros(1000, dtype=int)
for cat in categories:
    mask = fake_cls == cat
    thr  = cat_thresholds[cat]['threshold']
    preds_cat[mask] = (all_probs[mask] >= thr).astype(int)

overall_cat = (preds_cat == labels).mean()
print(f"\nOverall with per-category thresholds: {overall_cat:.4f}")
print(f"vs global best: {t1['best_overall']:.4f} | delta: {overall_cat - t1['best_overall']:+.4f}")

per_cat_results = {}
for cat in categories:
    mask = fake_cls == cat
    acc  = (preds_cat[mask] == labels[mask]).mean()
    per_cat_results[cat] = round(float(acc), 4)
    print(f"  {cat}: {acc:.4f}")

# ── Strategy 2: Grid search over all 4 thresholds jointly ──
# Too slow for fine grid — use coarse search first
print("\n=== Strategy 2: Joint Grid Search (coarse) ===")
coarse = np.arange(0.10, 0.71, 0.10)
best_joint_acc = 0
best_joint_thrs = {}

for thrs in product(coarse, repeat=4):
    preds_joint = np.zeros(1000, dtype=int)
    for i, cat in enumerate(categories):
        mask = fake_cls == cat
        preds_joint[mask] = (all_probs[mask] >= thrs[i]).astype(int)
    acc = (preds_joint == labels).mean()
    if acc > best_joint_acc:
        best_joint_acc = acc
        best_joint_thrs = dict(zip(categories, [round(float(t), 2) for t in thrs]))

print(f"Best joint: {best_joint_acc:.4f} with thresholds: {best_joint_thrs}")

# Fine search around best joint thresholds
print("\n=== Strategy 2: Fine Search Around Best ===")
fine_grids = []
for cat in categories:
    base = best_joint_thrs[cat]
    fine_grids.append(np.arange(max(0.05, base-0.10), min(0.95, base+0.11), 0.05))

best_fine_acc = best_joint_acc
best_fine_thrs = best_joint_thrs.copy()
for thrs in product(*fine_grids):
    preds_fine = np.zeros(1000, dtype=int)
    for i, cat in enumerate(categories):
        mask = fake_cls == cat
        preds_fine[mask] = (all_probs[mask] >= thrs[i]).astype(int)
    acc = (preds_fine == labels).mean()
    if acc > best_fine_acc:
        best_fine_acc = acc
        best_fine_thrs = dict(zip(categories, [round(float(t), 2) for t in thrs]))

print(f"Best fine: {best_fine_acc:.4f} with thresholds: {best_fine_thrs}")
print(f"Delta vs global best: {best_fine_acc - t1['best_overall']:+.4f}")

# Apply best fine thresholds and get per-category breakdown
preds_final = np.zeros(1000, dtype=int)
for cat in categories:
    mask = fake_cls == cat
    preds_final[mask] = (all_probs[mask] >= best_fine_thrs[cat]).astype(int)

final_per_cat = {}
for cat in categories:
    mask = fake_cls == cat
    acc  = (preds_final[mask] == labels[mask]).mean()
    final_per_cat[cat] = round(float(acc), 4)
    print(f"  {cat}: {acc:.4f}")

output = {
    'global_best_thr':         t1['best_threshold'],
    'global_best_acc':         t1['best_overall'],
    'independent_cat_thrs':    cat_thresholds,
    'independent_cat_overall': round(float(overall_cat), 4),
    'independent_cat_per_cat': per_cat_results,
    'joint_best_thrs':         best_fine_thrs,
    'joint_best_overall':      round(float(best_fine_acc), 4),
    'joint_best_per_cat':      final_per_cat,
    'delta_vs_global':         round(float(best_fine_acc - t1['best_overall']), 4),
}
with open(os.path.join(OUT_DIR, 'task2_per_cat_thresholds.json'), 'w') as f:
    json.dump(output, f, indent=2)
print("Task 2 DONE")
```

---

## Task 3 — Attention Explainability on MMFakeBench

**Goal:** Extract AITR attention weights using real MMFakeBench features. Per-category breakdown shows what the model focuses on for each fake type. This is your explainability contribution.

Save as `overnight_v2/task3_attention.py` and run.

```python
import torch, numpy as np, pandas as pd, json, os, csv
import torch.nn.functional as F

# paste AITRFusion, load_aitr_mmfb, paths, load_mmfb_val here

def forward_with_attention(model, img, txt, scalars):
    """Forward pass extracting CLS attention weights from layer 0."""
    B = img.size(0)
    img_times_txt = img * txt
    img_minus_txt = img - txt
    scalar_token  = model.scalar_proj(scalars)
    type_ids  = torch.arange(5, device=img.device)
    type_embs = model.type_embedding(type_ids)
    tokens = torch.cat([
        model.cls_token.expand(B, -1, -1),
        (img           + type_embs[0]).unsqueeze(1),
        (txt           + type_embs[1]).unsqueeze(1),
        (img_times_txt + type_embs[2]).unsqueeze(1),
        (img_minus_txt + type_embs[3]).unsqueeze(1),
        (scalar_token  + type_embs[4]).unsqueeze(1),
    ], dim=1).transpose(0, 1)   # (6, B, 768)

    # Extract attention from layer 0
    attn_layer = model.transformer.layers[0].self_attn
    try:
        _, attn_weights = attn_layer(tokens, tokens, tokens,
                                      need_weights=True,
                                      average_attn_weights=True)
        cls_attn = attn_weights[:, 0, 1:]  # (B, 5) CLS -> [img,txt,cross,diff,scalar]
    except Exception as e:
        print(f"Attention extraction error: {e} — using uniform fallback")
        cls_attn = torch.full((B, 5), 0.2)

    out    = model.transformer(tokens)
    logits = model.classifier(out[0]).squeeze(-1)
    return logits, cls_attn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data   = load_mmfb_val()
model  = load_aitr_mmfb(MMFB_AITR_PATH, device)

img     = data['img'].to(device)
txt     = data['txt'].to(device)
scalars = data['scalars'].to(device)
labels  = data['labels']
fake_cls = data['fake_cls']
categories = data['categories']
token_names = ['img', 'txt', 'cross', 'diff', 'scalars']

with open(os.path.join(OUT_DIR, 'task1_baseline.json')) as f:
    best_thr = json.load(f)['best_threshold']

all_logits, all_attn = [], []
with torch.no_grad():
    for i in range(0, 1000, 64):
        logits, attn = forward_with_attention(
            model, img[i:i+64], txt[i:i+64], scalars[i:i+64])
        all_logits.append(logits.cpu())
        all_attn.append(attn.cpu())
        print(f"  Batch {i//64 + 1}/16 done")

all_logits = torch.cat(all_logits).numpy()
all_attn   = torch.cat(all_attn).numpy()   # (1000, 5)
all_probs  = 1 / (1 + np.exp(-all_logits))
preds      = (all_probs >= best_thr).astype(int)
correct    = preds == labels

print("\n=== Mean CLS Attention — All Samples ===")
for i, name in enumerate(token_names):
    print(f"  {name}: {all_attn[:, i].mean():.4f}")

print("\n=== Mean CLS Attention — Per Category ===")
cat_attn = {}
for cat in categories:
    mask = fake_cls == cat
    cat_attn[cat] = {name: round(float(all_attn[mask, i].mean()), 4)
                     for i, name in enumerate(token_names)}
    print(f"  {cat}:")
    for name, val in cat_attn[cat].items():
        print(f"    {name}: {val:.4f}")

print("\n=== Dominant Token Distribution ===")
dominant_idx   = np.argmax(all_attn, axis=1)
dominant_names = [token_names[i] for i in dominant_idx]
dom_overall = {}
for name in token_names:
    pct = (np.array(dominant_names) == name).mean() * 100
    dom_overall[name] = round(float(pct), 1)
    print(f"  {name}: {pct:.1f}%")

print("\n=== Dominant Token — Per Category ===")
dom_per_cat = {}
for cat in categories:
    mask = fake_cls == cat
    dom_cat = [dominant_names[i] for i in range(1000) if mask[i]]
    dom_per_cat[cat] = {name: round(float((np.array(dom_cat)==name).mean()*100), 1)
                        for name in token_names}
    print(f"  {cat}: {dom_per_cat[cat]}")

print("\n=== Correct vs Incorrect Attention ===")
for split, mask in [('correct', correct), ('incorrect', ~correct)]:
    print(f"  {split}:")
    for i, name in enumerate(token_names):
        print(f"    {name}: {all_attn[mask, i].mean():.4f}")

# Save per-sample CSV
rows = []
for idx in range(1000):
    rows.append({
        'sample_idx':     idx,
        'fake_cls':       fake_cls[idx],
        'label':          int(labels[idx]),
        'prediction':     int(preds[idx]),
        'correct':        int(correct[idx]),
        'dominant_token': dominant_names[idx],
        'attn_img':       round(float(all_attn[idx, 0]), 4),
        'attn_txt':       round(float(all_attn[idx, 1]), 4),
        'attn_cross':     round(float(all_attn[idx, 2]), 4),
        'attn_diff':      round(float(all_attn[idx, 3]), 4),
        'attn_scalars':   round(float(all_attn[idx, 4]), 4),
        'aitr_prob':      round(float(all_probs[idx]), 4),
        'clip_prob':      round(float(data['probs'][idx]), 4),
        'ateeq_score':    round(float(data['ateeq'][idx]), 4),
        'deberta_score':  round(float(data['deberta'][idx]), 4),
    })

with open(os.path.join(OUT_DIR, 'per_sample_attention_mmfb.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

attn_summary = {
    'mean_all':         {name: round(float(all_attn[:, i].mean()), 4)
                         for i, name in enumerate(token_names)},
    'mean_per_cat':     cat_attn,
    'dominant_overall': dom_overall,
    'dominant_per_cat': dom_per_cat,
}
with open(os.path.join(OUT_DIR, 'task3_attention.json'), 'w') as f:
    json.dump(attn_summary, f, indent=2)
print("Task 3 DONE")
```

---

## Task 4 — Soft Ensemble: Variant B + Variant C

**Goal:** Average probabilities from Variant B (clip+ateeq) and Variant C (clip+deberta). May recover D's intended benefit without D's instability.

Save as `overnight_v2/task4_ensemble.py` and run.

```python
import torch, numpy as np, pandas as pd, json, os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# paste AITRFusion, load_aitr_mmfb, paths, load_mmfb_val here

# NOTE: AITRFusion needs to handle different scalar_dim per variant
class AITRFusionFlex(AITRFusion):
    def __init__(self, num_scalars):
        super().__init__(num_scalars=num_scalars)

def load_variant(path, num_scalars, device):
    model = AITRFusion(num_scalars=num_scalars)
    ckpt  = torch.load(path, map_location=device)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    res   = model.load_state_dict(state, strict=True)
    print(f"Loaded {path} (scalars={num_scalars}): missing={res.missing_keys}")
    return model.to(device).eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data   = load_mmfb_val()
labels  = data['labels']
fake_cls = data['fake_cls']
categories = data['categories']

img = data['img'].to(device)
txt = data['txt'].to(device)

# Load Variant B (scalars=3: clip_prob, clip_sim, ateeq)
scalars_B = data['scalars'].to(device)   # already [clip_prob, clip_sim, ateeq]

# Load Variant C (scalars=3: clip_prob, clip_sim, deberta)
scalars_C = torch.tensor(
    np.stack([data['probs'], data['sims'], data['deberta']], axis=1),
    dtype=torch.float32
).to(device)

# Load Variant D (scalars=4: clip_prob, clip_sim, ateeq, deberta)
scalars_D = torch.tensor(
    np.stack([data['probs'], data['sims'], data['ateeq'], data['deberta']], axis=1),
    dtype=torch.float32
).to(device)

TRAIN_DIR = r'D:\Pics Can Lie\mmfakebench_training'
variant_paths = {
    'B': (os.path.join(TRAIN_DIR, 'aitr_mmfb_v2_B.pt'), 3, scalars_B),
    'C': (os.path.join(TRAIN_DIR, 'aitr_mmfb_v2_C.pt'), 3, scalars_C),
    'D': (os.path.join(TRAIN_DIR, 'aitr_mmfb_v2_D.pt'), 4, scalars_D),
}

variant_probs = {}
for vname, (vpath, n_scalars, vscalars) in variant_paths.items():
    if not os.path.exists(vpath):
        print(f"Variant {vname} checkpoint not found at {vpath} — skipping")
        continue
    try:
        vmodel = load_variant(vpath, n_scalars, device)
        probs_list = []
        with torch.no_grad():
            for i in range(0, 1000, 64):
                logits = vmodel(img[i:i+64], txt[i:i+64], vscalars[i:i+64])
                probs_list.append(torch.sigmoid(logits).cpu())
        variant_probs[vname] = torch.cat(probs_list).numpy()
        auc = roc_auc_score(labels, variant_probs[vname])
        best_acc = max(
            ((variant_probs[vname] >= thr).astype(int) == labels).mean()
            for thr in np.arange(0.05, 0.96, 0.05)
        )
        print(f"Variant {vname}: AUC={auc:.4f}, best_acc={best_acc:.4f}")
    except Exception as e:
        print(f"Variant {vname} failed: {e}")

# Soft ensembles
ensembles = {}
available = list(variant_probs.keys())
print(f"\nAvailable variants for ensemble: {available}")

ensemble_configs = []
if 'B' in variant_probs and 'C' in variant_probs:
    ensemble_configs.append(('B+C_equal', {'B': 0.5, 'C': 0.5}))
    ensemble_configs.append(('B+C_Bheavy', {'B': 0.7, 'C': 0.3}))
if 'B' in variant_probs and 'D' in variant_probs:
    ensemble_configs.append(('B+D_equal', {'B': 0.5, 'D': 0.5}))
if len(available) == 3:
    ensemble_configs.append(('B+C+D_equal', {'B': 0.333, 'C': 0.333, 'D': 0.333}))

results = {}
for ename, weights in ensemble_configs:
    ens_probs = sum(variant_probs[v] * w for v, w in weights.items() if v in variant_probs)
    auc = roc_auc_score(labels, ens_probs)
    best_acc, best_thr = 0, 0.30
    for thr in np.arange(0.05, 0.96, 0.05):
        acc = ((ens_probs >= thr).astype(int) == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    preds = (ens_probs >= best_thr).astype(int)
    per_cat = {cat: round(float((preds[fake_cls==cat]==labels[fake_cls==cat]).mean()), 4)
               for cat in categories}
    results[ename] = {
        'weights': weights, 'auc': round(float(auc), 4),
        'best_thr': round(float(best_thr), 2),
        'best_acc': round(float(best_acc), 4),
        'per_cat': per_cat,
    }
    print(f"\n{ename}: AUC={auc:.4f}, best={best_acc:.4f} @ thr={best_thr:.2f}")
    for cat, acc in per_cat.items():
        print(f"  {cat}: {acc:.4f}")

with open(os.path.join(OUT_DIR, 'task4_ensemble.json'), 'w') as f:
    json.dump({'variant_probs_available': available, 'ensembles': results}, f, indent=2)
print("Task 4 DONE")
```

---

## Task 5 — Wikipedia NLI Veto on `original` Category

**Goal:** Use deberta_score as a veto signal for the `original` category. If deberta entailment is high AND AITR is uncertain → predict REAL. Target: recover some of the 41% errors in `original`.

Save as `overnight_v2/task5_wiki_veto.py` and run.

```python
import torch, numpy as np, pandas as pd, json, os
import torch.nn.functional as F

# paste paths and load_mmfb_val here

data      = load_mmfb_val()
labels    = data['labels']
fake_cls  = data['fake_cls']
categories = data['categories']
deberta   = data['deberta']

with open(os.path.join(OUT_DIR, 'task1_baseline.json')) as f:
    t1 = json.load(f)
all_probs = np.array(t1['all_probs'])
base_thr  = t1['best_threshold']
base_preds = (all_probs >= base_thr).astype(int)
base_acc   = (base_preds == labels).mean()

print(f"Baseline overall: {base_acc:.4f}")
print(f"Baseline original: {(base_preds[fake_cls=='original']==labels[fake_cls=='original']).mean():.4f}")

# Veto strategy: for 'original' samples where AITR predicts FAKE
# but deberta_score > threshold → override to REAL
results = []
veto_thresholds    = np.arange(0.10, 0.91, 0.05)
uncertainty_bands  = [(0.0, 1.0), (0.3, 0.7), (0.2, 0.8), (0.35, 0.65)]

for deb_thr in veto_thresholds:
    for unc_lo, unc_hi in uncertainty_bands:
        veto_preds = base_preds.copy()
        orig_mask  = fake_cls == 'original'
        # Veto condition: predicted FAKE + deberta high + AITR uncertain
        veto_cond = (
            orig_mask &
            (base_preds == 1) &
            (deberta >= deb_thr) &
            (all_probs >= unc_lo) & (all_probs <= unc_hi)
        )
        veto_preds[veto_cond] = 0   # override to REAL
        n_vetoed = veto_cond.sum()
        if n_vetoed == 0:
            continue
        overall = (veto_preds == labels).mean()
        orig_acc = (veto_preds[orig_mask] == labels[orig_mask]).mean()
        delta    = overall - base_acc
        results.append({
            'deb_thr':    round(float(deb_thr), 2),
            'unc_band':   f"{unc_lo}-{unc_hi}",
            'n_vetoed':   int(n_vetoed),
            'overall':    round(float(overall), 4),
            'orig_acc':   round(float(orig_acc), 4),
            'delta':      round(float(delta), 4),
        })

results.sort(key=lambda x: x['overall'], reverse=True)
print("\nTop 10 veto configs:")
for r in results[:10]:
    print(f"  deb>{r['deb_thr']} unc={r['unc_band']} vetoed={r['n_vetoed']}: "
          f"overall={r['overall']:.4f} (Δ{r['delta']:+.4f}) orig={r['orig_acc']:.4f}")

best_veto = results[0] if results else {}
if best_veto:
    print(f"\nBest veto: {best_veto}")
    # Apply best veto and get full per-category breakdown
    veto_preds = base_preds.copy()
    unc_lo, unc_hi = map(float, best_veto['unc_band'].split('-'))
    orig_mask = fake_cls == 'original'
    veto_cond = (
        orig_mask &
        (base_preds == 1) &
        (deberta >= best_veto['deb_thr']) &
        (all_probs >= unc_lo) & (all_probs <= unc_hi)
    )
    veto_preds[veto_cond] = 0
    for cat in categories:
        mask = fake_cls == cat
        acc  = (veto_preds[mask] == labels[mask]).mean()
        print(f"  {cat}: {acc:.4f}")

output = {
    'baseline_overall': round(float(base_acc), 4),
    'best_veto':        best_veto,
    'all_configs':      results[:20],
}
with open(os.path.join(OUT_DIR, 'task5_wiki_veto.json'), 'w') as f:
    json.dump(output, f, indent=2)
print("Task 5 DONE")
```

---

## Task 6 — Write DONE_v2.md

Save as `overnight_v2/task6_done.py` and run.

```python
import json, os
from datetime import datetime

OUT_DIR = r'D:\Pics Can Lie\overnight_v2'

def load(fname):
    try:
        with open(os.path.join(OUT_DIR, fname)) as f:
            return json.load(f)
    except Exception as e:
        return {'error': str(e)}

t1  = load('task1_baseline.json')
t2  = load('task2_per_cat_thresholds.json')
t3  = load('task3_attention.json')
t4  = load('task4_ensemble.json')
t5  = load('task5_wiki_veto.json')

def fmt(v):
    return f"{v*100:.2f}%" if isinstance(v, float) else str(v)

lines = [
    "# Overnight Run v2 — DONE",
    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    "---",
    "",
    "## Task 1 — Baseline Reproduction",
]

if 'error' not in t1:
    lines += [
        f"AUC-ROC: {t1.get('auc_roc', 'N/A')}",
        f"Best threshold: {t1.get('best_threshold')} → Overall: {fmt(t1.get('best_overall', 0))}",
        f"At thr=0.30 (reported in thesis): {fmt(t1.get('at_030', {}).get('overall', 0))}",
        "Per-category at best threshold:",
    ] + [f"  {cat}: {fmt(v)}" for cat, v in t1.get('best_per_cat', {}).items()]
else:
    lines.append(f"FAILED: {t1['error']}")

lines += ["", "## Task 2 — Per-Category Threshold Optimization"]
if 'error' not in t2:
    lines += [
        f"Global best:              {fmt(t2.get('global_best_acc', 0))} @ thr={t2.get('global_best_thr')}",
        f"Independent cat thrs:     {fmt(t2.get('independent_cat_overall', 0))}",
        f"Joint optimized thrs:     {fmt(t2.get('joint_best_overall', 0))} (Δ{t2.get('delta_vs_global', 0):+.4f})",
        f"Best thresholds per cat:  {t2.get('joint_best_thrs')}",
        "Per-category (joint best):",
    ] + [f"  {cat}: {fmt(v)}" for cat, v in t2.get('joint_best_per_cat', {}).items()]
else:
    lines.append(f"FAILED: {t2['error']}")

lines += ["", "## Task 3 — Attention Explainability"]
if 'error' not in t3:
    lines.append("Mean CLS attention weights (all samples):")
    for tok, val in t3.get('mean_all', {}).items():
        lines.append(f"  {tok}: {val:.4f}")
    lines.append("Dominant token per category:")
    for cat, dom in t3.get('dominant_per_cat', {}).items():
        top = max(dom, key=dom.get)
        lines.append(f"  {cat}: {top} ({dom[top]}%)")
else:
    lines.append(f"FAILED: {t3['error']}")

lines += ["", "## Task 4 — Soft Ensemble"]
if 'error' not in t4:
    for ename, res in t4.get('ensembles', {}).items():
        lines.append(f"  {ename}: {fmt(res['best_acc'])} @ thr={res['best_thr']} AUC={res['auc']}")
else:
    lines.append(f"FAILED: {t4['error']}")

lines += ["", "## Task 5 — Wikipedia Veto on original"]
if 'error' not in t5:
    bv = t5.get('best_veto', {})
    lines += [
        f"Baseline original accuracy: see Task 1",
        f"Best veto config: deb>{bv.get('deb_thr')} unc={bv.get('unc_band')} vetoed={bv.get('n_vetoed')}",
        f"Overall with veto: {fmt(bv.get('overall', 0))} (Δ{bv.get('delta', 0):+.4f})",
        f"Original accuracy with veto: {fmt(bv.get('orig_acc', 0))}",
    ]
else:
    lines.append(f"FAILED: {t5['error']}")

lines += [
    "",
    "## Summary Table — MMFakeBench Overall Accuracy",
    f"| System                          | Overall  |",
    f"|---|---|",
    f"| Baseline (global thr)           | {fmt(t1.get('best_overall', 0))} |",
    f"| + Per-category thresholds       | {fmt(t2.get('joint_best_overall', 0))} |",
    f"| + Wiki veto on original         | {fmt(t5.get('best_veto', {}).get('overall', 0))} |",
    "",
    "## Files Written",
    "- task1_baseline.json",
    "- task2_per_cat_thresholds.json",
    "- task3_attention.json + per_sample_attention_mmfb.csv",
    "- task4_ensemble.json",
    "- task5_wiki_veto.json",
    "",
    "## Key Questions for Morning",
    "1. Did per-category thresholds beat 72.6%? By how much?",
    "2. Which token dominates per category? (scalars for visual_vd expected)",
    "3. Did any ensemble beat Variant B?",
    "4. Did wiki veto recover original accuracy meaningfully?",
    "5. What is the new headline number for MMFakeBench?",
]

done_text = "\n".join(lines)
with open(os.path.join(OUT_DIR, 'DONE_v2.md'), 'w') as f:
    f.write(done_text)
print(done_text)
print("\n" + "="*50)
print("ALL TASKS COMPLETE — check DONE_v2.md")
print("="*50)
```

---

## Final Instructions to Claude Code

1. Create `D:\Pics Can Lie\overnight_v2\` if it doesn't exist
2. For each task: save the script first, then run with `.\venv\Scripts\python.exe`
3. Every script needs the AITRFusion class + load_aitr_mmfb + file paths + load_mmfb_val pasted at the top
4. Run in order: Task 1 → 2 → 3 → 4 → 5 → 6
5. Task 4 depends on variant checkpoint files in `mmfakebench_training/` — if they don't exist, skip gracefully
6. If any task fails, write the error and continue to next task
7. Do NOT retrain anything
8. Do NOT touch files outside `overnight_v2\`
9. When complete: print **"ALL TASKS COMPLETE — check DONE_v2.md"**
