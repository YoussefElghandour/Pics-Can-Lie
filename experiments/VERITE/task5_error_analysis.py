import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json

df     = pd.read_csv(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'VERITE.csv'))
labels = (df['label'] != 'true').astype(int).values

with open(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'aitr_fusion_results.json')) as f:
    task3     = json.load(f)
all_probs = np.array(task3['all_probs'])
best_thr  = task3['best_threshold']
preds     = (all_probs >= best_thr).astype(int)
correct   = preds == labels

fp_mask = (preds == 1) & (labels == 0)
fn_mask = (preds == 0) & (labels == 1)
print(f"Total errors: {(~correct).sum()} / {len(labels)}")
print(f"FP (real->fake): {fp_mask.sum()}")
print(f"FN (fake->real): {fn_mask.sum()}")
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

img   = np.load(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'VERITE_clip_image_embeddings_ViTL14.npy'))
txt   = np.load(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'VERITE_clip_text_embeddings_ViTL14.npy'))
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
with open(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'error_analysis.json'), 'w') as f:
    json.dump(output, f, indent=2)
print("Task 5 DONE - saved error_analysis.json")
