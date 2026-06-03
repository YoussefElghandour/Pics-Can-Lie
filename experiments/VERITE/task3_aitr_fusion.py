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
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import json
from _aitr import load_aitr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

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
with open(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'aitr_fusion_results.json'), 'w') as f:
    json.dump(output, f, indent=2)
print("Task 3 DONE — saved aitr_fusion_results.json")
