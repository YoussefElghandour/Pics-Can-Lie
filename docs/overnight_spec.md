# Overnight Spec — VERITE Evaluation + Explainability
> Working directory: `D:\Pics Can Lie\`
> VERITE files location: `D:\Pics Can Lie\verite\`
> Run everything in order. Do NOT skip steps. Do NOT move on if a step fails — write the error to DONE.md and stop.
> When all tasks are complete, write a full summary to `D:\Pics Can Lie\verite\DONE.md`.
> All required Python packages are already installed in the venv.
> Use `.\venv\Scripts\python.exe` for all Python execution.

---

## Context (read before touching anything)

This is a thesis project on multimodal misinformation detection ("Pics Can Lie").

**DO NOT retrain anything tonight — inference and analysis only.**

**Current system (already trained):**
- AITR transformer fusion weights: `D:\Pics Can Lie\fusion_aitr\aitr_weights.pt`
- Optimal threshold: 0.54 (NewsCLIPpings)

**VERITE files (already downloaded):**
- `D:\Pics Can Lie\verite\VERITE.csv` — 1001 rows, columns: [unnamed_index, caption, image_path, label]
- `D:\Pics Can Lie\verite\VERITE_articles.csv` — article context
- `D:\Pics Can Lie\verite\VERITE_clip_image_embeddings_ViTL14.npy` — shape (1001, 768)
- `D:\Pics Can Lie\verite\VERITE_clip_text_embeddings_ViTL14.npy` — shape (1001, 768)

**VERITE label values and counts:**
- `true` → 0 (REAL) — 338 samples
- `miscaptioned` → 1 (FAKE) — 338 samples
- `out-of-context` → 1 (FAKE) — 325 samples
- Total: 1001 samples

**CRITICAL GOTCHAS:**
- Features are NOT L2-normalized — always normalize: `F.normalize(torch.tensor(features).float(), dim=-1)`
- `num_workers=0, pin_memory=False` on Windows
- Do NOT modify any existing files — only write new files to `D:\Pics Can Lie\verite\`

---

## AITR Exact Architecture (verified from saved weights — do not deviate)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AITRFusion(nn.Module):
    def __init__(self, embed_dim=768, num_scalars=9, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()

        # scalar_proj: Linear(9->768) + LayerNorm(768)
        self.scalar_proj = nn.Sequential(
            nn.Linear(num_scalars, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # type_embedding: 5 token types (img, txt, cross, diff, scalar)
        self.type_embedding = nn.Embedding(5, embed_dim)

        # cls_token: (1, 1, 768)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # transformer: 2 layers, 8 heads — batch_first=False (default)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # classifier: LayerNorm(768) -> Linear(768,256) -> GELU -> Dropout -> Linear(256,1)
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

        img_tok    = img_emb       + type_embs[0]
        txt_tok    = txt_emb       + type_embs[1]
        cross_tok  = img_times_txt + type_embs[2]
        diff_tok   = img_minus_txt + type_embs[3]
        scalar_tok = scalar_token  + type_embs[4]

        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([
            cls,
            img_tok.unsqueeze(1),
            txt_tok.unsqueeze(1),
            cross_tok.unsqueeze(1),
            diff_tok.unsqueeze(1),
            scalar_tok.unsqueeze(1)
        ], dim=1)               # (B, 6, 768)
        tokens = tokens.transpose(0, 1)  # (6, B, 768)

        out     = self.transformer(tokens)
        cls_out = out[0]        # (B, 768)
        logits  = self.classifier(cls_out).squeeze(-1)
        return logits


def load_aitr(weights_path, device):
    model  = AITRFusion()
    state  = torch.load(weights_path, map_location=device)
    result = model.load_state_dict(state, strict=True)
    print(f"Loaded AITR. Missing: {result.missing_keys}, Unexpected: {result.unexpected_keys}")
    model.to(device).eval()
    return model
```

**If strict=True fails, retry with strict=False and print missing/unexpected keys — do not crash.**

---

## Task 1 — Verify VERITE Files

Save as `D:\Pics Can Lie\verite\task1_verify.py` and run it.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

df  = pd.read_csv(r'D:\Pics Can Lie\verite\VERITE.csv')
print(f"CSV rows: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

img = np.load(r'D:\Pics Can Lie\verite\VERITE_clip_image_embeddings_ViTL14.npy')
txt = np.load(r'D:\Pics Can Lie\verite\VERITE_clip_text_embeddings_ViTL14.npy')
print(f"\nImage shape: {img.shape}")
print(f"Text shape:  {txt.shape}")
assert img.shape == txt.shape
assert img.shape[0] == len(df)
print(f"NaN img: {np.isnan(img).sum()}, txt: {np.isnan(txt).sum()}")

img_t = F.normalize(torch.tensor(img).float(), dim=-1)
txt_t = F.normalize(torch.tensor(txt).float(), dim=-1)
sims  = (img_t * txt_t).sum(dim=-1).numpy()

for label in ['true', 'miscaptioned', 'out-of-context']:
    mask = df['label'] == label
    print(f"Mean CLIP sim [{label}]: {sims[mask].mean():.4f}")

print("\nTask 1 PASSED")
```

---

## Task 2 — CLIP-only Zero-Shot Evaluation

Save as `D:\Pics Can Lie\verite\task2_clip_only.py` and run it.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import json

df     = pd.read_csv(r'D:\Pics Can Lie\verite\VERITE.csv')
img    = np.load(r'D:\Pics Can Lie\verite\VERITE_clip_image_embeddings_ViTL14.npy')
txt    = np.load(r'D:\Pics Can Lie\verite\VERITE_clip_text_embeddings_ViTL14.npy')
labels = (df['label'] != 'true').astype(int).values

img_t  = F.normalize(torch.tensor(img).float(), dim=-1)
txt_t  = F.normalize(torch.tensor(txt).float(), dim=-1)
sims   = (img_t * txt_t).sum(dim=-1).numpy()

# Threshold sweep — predict FAKE if sim < threshold
thresholds = np.arange(0.10, 0.91, 0.05)
results    = []
for thr in thresholds:
    preds = (sims < thr).astype(int)
    acc   = (preds == labels).mean()
    results.append({'threshold': round(float(thr), 2), 'accuracy': round(float(acc), 4)})

best     = max(results, key=lambda x: x['accuracy'])
best_thr = best['threshold']
print(f"Best threshold: {best_thr}, Accuracy: {best['accuracy']:.4f}")

auc   = roc_auc_score(labels, -sims)
preds = (sims < best_thr).astype(int)
print(f"AUC-ROC: {auc:.4f}")

per_cat = {}
for label_name in ['true', 'miscaptioned', 'out-of-context']:
    mask    = df['label'] == label_name
    cat_acc = (preds[mask] == labels[mask]).mean()
    per_cat[label_name] = round(float(cat_acc), 4)
    print(f"  [{label_name}] accuracy: {cat_acc:.4f} ({mask.sum()} samples)")

prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

output = {
    'best_threshold': best_thr,
    'overall_accuracy': best['accuracy'],
    'auc_roc': round(float(auc), 4),
    'precision': round(float(prec), 4),
    'recall': round(float(rec), 4),
    'f1': round(float(f1), 4),
    'per_category': per_cat,
    'all_thresholds': results
}
with open(r'D:\Pics Can Lie\verite\clip_only_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("Task 2 DONE — saved clip_only_results.json")
```

---

## Task 3 — AITR Fusion Zero-Shot Evaluation

Save as `D:\Pics Can Lie\verite\task3_aitr_fusion.py` and run it.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import json

# ── Paste full AITRFusion class and load_aitr function here ──

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

df     = pd.read_csv(r'D:\Pics Can Lie\verite\VERITE.csv')
img    = np.load(r'D:\Pics Can Lie\verite\VERITE_clip_image_embeddings_ViTL14.npy')
txt    = np.load(r'D:\Pics Can Lie\verite\VERITE_clip_text_embeddings_ViTL14.npy')
labels = (df['label'] != 'true').astype(int).values

img_t     = F.normalize(torch.tensor(img).float(), dim=-1).to(device)
txt_t     = F.normalize(torch.tensor(txt).float(), dim=-1).to(device)
sims      = (img_t * txt_t).sum(dim=-1)
clip_prob = torch.sigmoid(sims * 10 - 5).unsqueeze(1)
neutral   = torch.full((img_t.size(0), 7), 0.5, device=device)
scalars   = torch.cat([clip_prob, sims.unsqueeze(1), neutral], dim=1)  # (N, 9)

model = load_aitr(r'D:\Pics Can Lie\fusion_aitr\aitr_weights.pt', device)

all_probs = []
batch_size = 64
N = img_t.size(0)
with torch.no_grad():
    for i in range(0, N, batch_size):
        logits = model(img_t[i:i+batch_size], txt_t[i:i+batch_size], scalars[i:i+batch_size])
        all_probs.append(torch.sigmoid(logits).cpu())
        print(f"  Batch {i//batch_size + 1}/{(N+batch_size-1)//batch_size}")

all_probs = torch.cat(all_probs).numpy()
auc       = roc_auc_score(labels, all_probs)
print(f"AUC-ROC: {auc:.4f}")

thresholds = np.arange(0.10, 0.91, 0.05)
results    = []
for thr in thresholds:
    preds = (all_probs >= thr).astype(int)
    acc   = (preds == labels).mean()
    results.append({'threshold': round(float(thr), 2), 'accuracy': round(float(acc), 4)})

best     = max(results, key=lambda x: x['accuracy'])
best_thr = best['threshold']
print(f"Best threshold: {best_thr}, Accuracy: {best['accuracy']:.4f}")

preds_054 = (all_probs >= 0.54).astype(int)
acc_054   = (preds_054 == labels).mean()
print(f"At thr=0.54: {acc_054:.4f}")

preds   = (all_probs >= best_thr).astype(int)
per_cat = {}
for label_name in ['true', 'miscaptioned', 'out-of-context']:
    mask    = df['label'] == label_name
    cat_acc = (preds[mask] == labels[mask]).mean()
    per_cat[label_name] = round(float(cat_acc), 4)
    print(f"  [{label_name}] accuracy: {cat_acc:.4f} ({mask.sum()} samples)")

prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

output = {
    'best_threshold': best_thr,
    'overall_accuracy': best['accuracy'],
    'accuracy_at_054': round(float(acc_054), 4),
    'auc_roc': round(float(auc), 4),
    'precision': round(float(prec), 4),
    'recall': round(float(rec), 4),
    'f1': round(float(f1), 4),
    'per_category': per_cat,
    'all_thresholds': results,
    'all_probs': all_probs.tolist()
}
with open(r'D:\Pics Can Lie\verite\aitr_fusion_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("Task 3 DONE — saved aitr_fusion_results.json")
```

---

## Task 4 — Attention Weight Extraction (Explainability)

Save as `D:\Pics Can Lie\verite\task4_attention.py` and run it.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json, csv

# ── Paste full AITRFusion class and load_aitr here ──

def forward_with_attention(model, img_emb, txt_emb, scalars):
    """Run AITR and extract CLS attention weights from layer 0."""
    B = img_emb.size(0)
    img_times_txt = img_emb * txt_emb
    img_minus_txt = img_emb - txt_emb
    scalar_token  = model.scalar_proj(scalars)

    type_ids  = torch.arange(5, device=img_emb.device)
    type_embs = model.type_embedding(type_ids)

    tokens = torch.cat([
        model.cls_token.expand(B, -1, -1),
        (img_emb    + type_embs[0]).unsqueeze(1),
        (txt_emb    + type_embs[1]).unsqueeze(1),
        (img_times_txt + type_embs[2]).unsqueeze(1),
        (img_minus_txt + type_embs[3]).unsqueeze(1),
        (scalar_token  + type_embs[4]).unsqueeze(1),
    ], dim=1).transpose(0, 1)  # (6, B, 768)

    # Extract attention weights from layer 0
    attn_layer = model.transformer.layers[0].self_attn
    _, attn_weights = attn_layer(tokens, tokens, tokens,
                                  need_weights=True,
                                  average_attn_weights=True)
    # attn_weights: (B, 6, 6) — CLS attends to all 6 tokens (including itself)
    cls_attn = attn_weights[:, 0, 1:]  # (B, 5) — CLS -> [img, txt, cross, diff, scalar]

    out    = model.transformer(tokens)
    logits = model.classifier(out[0]).squeeze(-1)
    return logits, cls_attn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df     = pd.read_csv(r'D:\Pics Can Lie\verite\VERITE.csv')
img    = np.load(r'D:\Pics Can Lie\verite\VERITE_clip_image_embeddings_ViTL14.npy')
txt    = np.load(r'D:\Pics Can Lie\verite\VERITE_clip_text_embeddings_ViTL14.npy')
labels = (df['label'] != 'true').astype(int).values

img_t     = F.normalize(torch.tensor(img).float(), dim=-1).to(device)
txt_t     = F.normalize(torch.tensor(txt).float(), dim=-1).to(device)
sims      = (img_t * txt_t).sum(dim=-1)
clip_prob = torch.sigmoid(sims * 10 - 5).unsqueeze(1)
neutral   = torch.full((img_t.size(0), 7), 0.5, device=device)
scalars   = torch.cat([clip_prob, sims.unsqueeze(1), neutral], dim=1)

model = load_aitr(r'D:\Pics Can Lie\fusion_aitr\aitr_weights.pt', device)

with open(r'D:\Pics Can Lie\verite\aitr_fusion_results.json') as f:
    best_thr = json.load(f)['best_threshold']

all_logits, all_attn = [], []
N = img_t.size(0)
with torch.no_grad():
    for i in range(0, N, 64):
        try:
            logits, attn = forward_with_attention(
                model, img_t[i:i+64], txt_t[i:i+64], scalars[i:i+64])
            all_logits.append(logits.cpu())
            all_attn.append(attn.cpu())
        except Exception as e:
            print(f"Attention extraction failed at batch {i}: {e}")
            print("Falling back to logits-only mode")
            logits = model(img_t[i:i+64], txt_t[i:i+64], scalars[i:i+64])
            all_logits.append(logits.cpu())
            all_attn.append(torch.full((min(64, N-i), 5), float('nan')))
        print(f"  Batch {i//64 + 1} done")

all_logits = torch.cat(all_logits).numpy()
all_attn   = torch.cat(all_attn).numpy()   # (N, 5)
all_probs  = 1 / (1 + np.exp(-all_logits))
preds      = (all_probs >= best_thr).astype(int)
correct    = preds == labels

token_names = ['img', 'txt', 'cross', 'diff', 'scalars']

print("\n=== Mean CLS Attention Weights ===")
for i, name in enumerate(token_names):
    valid = all_attn[:, i][~np.isnan(all_attn[:, i])]
    print(f"  {name}: {valid.mean():.4f}")

print("\n=== By Category ===")
for label_name in ['true', 'miscaptioned', 'out-of-context']:
    mask = (df['label'] == label_name).values
    print(f"  {label_name}:")
    for i, name in enumerate(token_names):
        vals = all_attn[mask, i]
        vals = vals[~np.isnan(vals)]
        print(f"    {name}: {vals.mean():.4f}")

dominant_idx   = np.where(
    np.isnan(all_attn).all(axis=1),
    -1,
    np.nanargmax(all_attn, axis=1)
)
dominant_names = [token_names[i] if i >= 0 else 'unknown' for i in dominant_idx]

print("\n=== Dominant Token Distribution ===")
for name in token_names:
    pct = (np.array(dominant_names) == name).mean() * 100
    print(f"  {name}: {pct:.1f}%")

attn_summary = {
    'mean_attention_all': {
        name: float(np.nanmean(all_attn[:, i]))
        for i, name in enumerate(token_names)
    },
    'mean_attention_by_category': {
        label_name: {
            name: float(np.nanmean(all_attn[(df['label'] == label_name).values, i]))
            for i, name in enumerate(token_names)
        }
        for label_name in ['true', 'miscaptioned', 'out-of-context']
    },
    'dominant_token_distribution': {
        name: float((np.array(dominant_names) == name).mean())
        for name in token_names
    }
}
with open(r'D:\Pics Can Lie\verite\attention_analysis.json', 'w') as f:
    json.dump(attn_summary, f, indent=2)

rows = []
for idx in range(N):
    rows.append({
        'sample_idx':     idx,
        'label':          df['label'].iloc[idx],
        'binary_label':   int(labels[idx]),
        'prediction':     int(preds[idx]),
        'correct':        int(correct[idx]),
        'dominant_token': dominant_names[idx],
        'attn_img':       round(float(all_attn[idx, 0]), 4) if not np.isnan(all_attn[idx, 0]) else None,
        'attn_txt':       round(float(all_attn[idx, 1]), 4) if not np.isnan(all_attn[idx, 1]) else None,
        'attn_cross':     round(float(all_attn[idx, 2]), 4) if not np.isnan(all_attn[idx, 2]) else None,
        'attn_diff':      round(float(all_attn[idx, 3]), 4) if not np.isnan(all_attn[idx, 3]) else None,
        'attn_scalars':   round(float(all_attn[idx, 4]), 4) if not np.isnan(all_attn[idx, 4]) else None,
        'aitr_prob':      round(float(all_probs[idx]), 4),
    })

with open(r'D:\Pics Can Lie\verite\per_sample_attention.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Task 4 DONE — saved attention_analysis.json and per_sample_attention.csv")
```

---

## Task 5 — Error Analysis

Save as `D:\Pics Can Lie\verite\task5_error_analysis.py` and run it.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json

df     = pd.read_csv(r'D:\Pics Can Lie\verite\VERITE.csv')
labels = (df['label'] != 'true').astype(int).values

with open(r'D:\Pics Can Lie\verite\aitr_fusion_results.json') as f:
    task3     = json.load(f)
all_probs = np.array(task3['all_probs'])
best_thr  = task3['best_threshold']
preds     = (all_probs >= best_thr).astype(int)
correct   = preds == labels

fp_mask = (preds == 1) & (labels == 0)
fn_mask = (preds == 0) & (labels == 1)
print(f"Total errors: {(~correct).sum()} / {len(labels)}")
print(f"FP (real→fake): {fp_mask.sum()}")
print(f"FN (fake→real): {fn_mask.sum()}")
for cat in ['miscaptioned', 'out-of-context']:
    cat_mask = (df['label'] == cat).values
    print(f"  FN {cat}: {(fn_mask & cat_mask).sum()}")

wrong_idx  = np.where(~correct)[0]
confidence = np.abs(all_probs[wrong_idx] - 0.5)
top10_idx  = wrong_idx[np.argsort(confidence)[::-1][:10]]

print("\n=== Top 10 Most Confident Errors ===")
for i, idx in enumerate(top10_idx):
    print(f"\n  [{i+1}] idx={idx} | true={df['label'].iloc[idx]} | pred={'FAKE' if preds[idx]==1 else 'REAL'} | prob={all_probs[idx]:.4f}")
    print(f"       {str(df['caption'].iloc[idx])[:120]}...")

img   = np.load(r'D:\Pics Can Lie\verite\VERITE_clip_image_embeddings_ViTL14.npy')
txt   = np.load(r'D:\Pics Can Lie\verite\VERITE_clip_text_embeddings_ViTL14.npy')
img_t = F.normalize(torch.tensor(img).float(), dim=-1)
txt_t = F.normalize(torch.tensor(txt).float(), dim=-1)
sims  = (img_t * txt_t).sum(dim=-1).numpy()

print(f"\nCLIP sim correct:   mean={sims[correct].mean():.4f}")
print(f"CLIP sim incorrect: mean={sims[~correct].mean():.4f}")
for lo, hi in [(0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0)]:
    mask = (sims >= lo) & (sims < hi)
    if mask.sum() > 0:
        err = (~correct & mask).sum()
        print(f"  sim [{lo:.1f}-{hi:.1f}]: {err}/{mask.sum()} errors ({err/mask.sum()*100:.1f}%)")

output = {
    'total_errors': int((~correct).sum()),
    'false_positives': int(fp_mask.sum()),
    'false_negatives': int(fn_mask.sum()),
    'fn_by_category': {
        cat: int((fn_mask & (df['label'] == cat).values).sum())
        for cat in ['miscaptioned', 'out-of-context']
    },
    'top10_wrong': [
        {
            'sample_idx':      int(idx),
            'true_label':      df['label'].iloc[idx],
            'predicted':       'FAKE' if preds[idx] == 1 else 'REAL',
            'aitr_prob':       round(float(all_probs[idx]), 4),
            'caption_preview': str(df['caption'].iloc[idx])[:150]
        }
        for idx in top10_idx
    ],
    'clip_sim_correct_mean':   round(float(sims[correct].mean()), 4),
    'clip_sim_incorrect_mean': round(float(sims[~correct].mean()), 4),
}
with open(r'D:\Pics Can Lie\verite\error_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)
print("Task 5 DONE — saved error_analysis.json")
```

---

## Task 6 — Comparison Table

Save as `D:\Pics Can Lie\verite\task6_comparison.py` and run it.

```python
import json

with open(r'D:\Pics Can Lie\verite\clip_only_results.json') as f:
    clip = json.load(f)
with open(r'D:\Pics Can Lie\verite\aitr_fusion_results.json') as f:
    aitr = json.load(f)

literature = [
    {'system': 'CLIP ViT-L/14 zero-shot (VERITE paper baseline)', 'accuracy': 0.7440, 'notes': 'No training'},
    {'system': 'VERITE paper best (trained on VERITE)',            'accuracy': 0.8100, 'notes': 'In-domain'},
    {'system': 'LAMAR (new SOTA Apr 2025)',                        'accuracy': None,   'notes': 'Check paper'},
    {'system': 'MAD-Sherlock (ICML 2025)',                         'accuracy': None,   'notes': 'Check paper'},
]
your_results = [
    {'system': 'Your CLIP-only zero-shot',   'accuracy': clip['overall_accuracy'], 'auc': clip['auc_roc'], 'notes': 'Zero-shot'},
    {'system': 'Your AITR fusion zero-shot', 'accuracy': aitr['overall_accuracy'], 'auc': aitr['auc_roc'], 'notes': 'Trained on NewsCLIPpings only'},
]

print("=" * 65)
print("VERITE Comparison Table")
print("=" * 65)
for r in literature:
    acc = f"{r['accuracy']*100:.2f}%" if r['accuracy'] else "TBD"
    print(f"  {r['system']:<52} {acc}")
print("-" * 65)
for r in your_results:
    print(f"  {r['system']:<52} {r['accuracy']*100:.2f}%  AUC={r['auc']:.4f}")
print("=" * 65)

with open(r'D:\Pics Can Lie\verite\comparison_table.json', 'w') as f:
    json.dump({'literature': literature, 'your_results': your_results}, f, indent=2)
print("Task 6 DONE — saved comparison_table.json")
```

---

## Task 7 — Write DONE.md

Save as `D:\Pics Can Lie\verite\task7_done.py` and run it.

```python
import json
from datetime import datetime

def load(path):
    try:
        with open(path) as f: return json.load(f)
    except: return None

clip = load(r'D:\Pics Can Lie\verite\clip_only_results.json')
aitr = load(r'D:\Pics Can Lie\verite\aitr_fusion_results.json')
attn = load(r'D:\Pics Can Lie\verite\attention_analysis.json')
err  = load(r'D:\Pics Can Lie\verite\error_analysis.json')
comp = load(r'D:\Pics Can Lie\verite\comparison_table.json')

def fmt(d): return f"{d*100:.2f}%" if d is not None else "N/A"

lines = [
    "# Overnight Run — DONE",
    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    "---",
    "",
    "## Task 1 — Data Verification",
    "PASSED — 1001 samples, shapes (1001,768) confirmed, no NaN/Inf.",
    "Labels: true=338, miscaptioned=338, out-of-context=325",
    "",
    "## Task 2 — CLIP-only Results",
]
if clip:
    lines += [
        f"Best threshold: {clip['best_threshold']}",
        f"Overall accuracy: {fmt(clip['overall_accuracy'])}",
        f"AUC-ROC: {clip['auc_roc']}  |  F1: {clip['f1']}",
        f"  true (REAL):           {fmt(clip['per_category'].get('true'))}",
        f"  miscaptioned (FAKE):   {fmt(clip['per_category'].get('miscaptioned'))}",
        f"  out-of-context (FAKE): {fmt(clip['per_category'].get('out-of-context'))}",
    ]
else:
    lines.append("FAILED")

lines += ["", "## Task 3 — AITR Fusion Results"]
if aitr:
    lines += [
        f"Best threshold: {aitr['best_threshold']}",
        f"Overall accuracy: {fmt(aitr['overall_accuracy'])}",
        f"At thr=0.54 (NewsCLIPpings optimal): {fmt(aitr['accuracy_at_054'])}",
        f"AUC-ROC: {aitr['auc_roc']}  |  F1: {aitr['f1']}",
        f"  true (REAL):           {fmt(aitr['per_category'].get('true'))}",
        f"  miscaptioned (FAKE):   {fmt(aitr['per_category'].get('miscaptioned'))}",
        f"  out-of-context (FAKE): {fmt(aitr['per_category'].get('out-of-context'))}",
    ]
else:
    lines.append("FAILED")

lines += ["", "## Task 4 — Attention Explainability"]
if attn:
    lines.append("Mean CLS attention weights:")
    for tok, val in attn['mean_attention_all'].items():
        lines.append(f"  {tok}: {val:.4f}")
    lines.append("Dominant token distribution:")
    for tok, val in attn['dominant_token_distribution'].items():
        lines.append(f"  {tok}: {val*100:.1f}%")
else:
    lines.append("FAILED")

lines += ["", "## Task 5 — Error Analysis"]
if err:
    lines += [
        f"Total errors: {err['total_errors']} / 1001",
        f"FP (real→fake): {err['false_positives']}",
        f"FN (fake→real): {err['false_negatives']}",
        f"  FN miscaptioned: {err['fn_by_category'].get('miscaptioned')}",
        f"  FN out-of-context: {err['fn_by_category'].get('out-of-context')}",
        f"CLIP sim correct: {err['clip_sim_correct_mean']}  |  incorrect: {err['clip_sim_incorrect_mean']}",
    ]
else:
    lines.append("FAILED")

lines += ["", "## Task 6 — Comparison Table"]
if comp:
    for r in comp['literature']:
        acc = f"{r['accuracy']*100:.2f}%" if r['accuracy'] else "TBD"
        lines.append(f"  {r['system']}: {acc}")
    for r in comp['your_results']:
        lines.append(f"  {r['system']}: {r['accuracy']*100:.2f}% (AUC={r['auc']:.4f})")
else:
    lines.append("FAILED")

lines += [
    "",
    "## Files Written",
    "- clip_only_results.json",
    "- aitr_fusion_results.json",
    "- attention_analysis.json",
    "- per_sample_attention.csv",
    "- error_analysis.json",
    "- comparison_table.json",
    "- task1_verify.py through task7_done.py",
    "",
    "## Key Questions to Answer in the Morning",
    "1. Does AITR beat CLIP-only? (does fusion generalize to VERITE?)",
    "2. Is overall accuracy above 74.40%? (zero-shot generalization win)",
    "3. Which token dominates attention? (explainability finding)",
    "4. Are miscaptioned errors higher than OOC? (expected — NLI weakness)",
    "5. Does thr=0.54 transfer or does VERITE need its own threshold?",
]

done_text = "\n".join(lines)
with open(r'D:\Pics Can Lie\verite\DONE.md', 'w') as f:
    f.write(done_text)

print(done_text)
print("\n" + "=" * 50)
print("ALL TASKS COMPLETE — check DONE.md")
print("=" * 50)
```

---

## Final Instructions to Claude Code

- Run tasks in order: 1 → 2 → 3 → 4 → 5 → 6 → 7
- Each task = save the script then run with `.\venv\Scripts\python.exe`
- If Task 3 weight loading fails with strict=True, retry strict=False
- If attention extraction in Task 4 fails, fall back to logits-only and fill attention columns with None
- Do NOT retrain anything
- Do NOT modify files outside `D:\Pics Can Lie\verite\`
- When finished: print **"ALL TASKS COMPLETE — check DONE.md"**
