import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json, csv
from _aitr import load_aitr


def forward_with_attention(model, img_emb, txt_emb, scalars):
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

    attn_layer = model.transformer.layers[0].self_attn
    _, attn_weights = attn_layer(tokens, tokens, tokens,
                                  need_weights=True,
                                  average_attn_weights=True)
    cls_attn = attn_weights[:, 0, 1:]  # (B, 5)

    out    = model.transformer(tokens)
    logits = model.classifier(out[0]).squeeze(-1)
    return logits, cls_attn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df     = pd.read_csv(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'VERITE.csv'))
img    = np.load(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'VERITE_clip_image_embeddings_ViTL14.npy'))
txt    = np.load(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'VERITE_clip_text_embeddings_ViTL14.npy'))
labels = (df['label'] != 'true').astype(int).values

img_t     = F.normalize(torch.tensor(img).float(), dim=-1).to(device)
txt_t     = F.normalize(torch.tensor(txt).float(), dim=-1).to(device)
sims      = (img_t * txt_t).sum(dim=-1)
clip_prob = torch.sigmoid(sims * 10 - 5).unsqueeze(1)
neutral   = torch.full((img_t.size(0), 7), 0.5, device=device)
scalars   = torch.cat([clip_prob, sims.unsqueeze(1), neutral], dim=1)

model = load_aitr(_os.path.join(str(_cfg.ROOT), 'models', 'fusion_aitr', 'aitr_weights.pt'), device)

with open(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'aitr_fusion_results.json')) as f:
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
all_attn   = torch.cat(all_attn).numpy()
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
with open(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'attention_analysis.json'), 'w') as f:
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

with open(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'per_sample_attention.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Task 4 DONE — saved attention_analysis.json and per_sample_attention.csv")
