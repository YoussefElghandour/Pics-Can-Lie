import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os


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
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=False
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
        ], dim=1).transpose(0, 1)
        out    = self.transformer(tokens)
        cls_out = out[0]
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


MMFB_AITR_PATH  = _os.path.join(str(_cfg.ROOT), 'experiments', 'mmfakebench_training', 'aitr_mmfb_best.pt')
VAL_IMG_PT      = _os.path.join(str(_cfg.ROOT), 'experiments', 'mmfakebench_training', 'val_features', 'clip_img.pt')
VAL_TXT_PT      = _os.path.join(str(_cfg.ROOT), 'experiments', 'mmfakebench_training', 'val_features', 'clip_txt.pt')
VAL_PROBS_NPY   = _os.path.join(str(_cfg.ROOT), 'experiments', 'mmfakebench_training', 'val_features', 'clip_probs.npy')
VAL_SIMS_NPY    = _os.path.join(str(_cfg.ROOT), 'experiments', 'mmfakebench_training', 'val_features', 'clip_sims.npy')
VAL_IDS_CSV     = _os.path.join(str(_cfg.ROOT), 'experiments', 'mmfakebench_training', 'val_features', 'sample_ids.csv')
ATEEQ_CSV       = _os.path.join(str(_cfg.ROOT), 'experiments', 'mmfakebench_training', 'val_ateeq_scores_full.csv')
DEBERTA_CSV     = _os.path.join(str(_cfg.ROOT), 'experiments', 'mmfakebench_training', 'deberta_nli_val.csv')
MMFB_VAL_JSON   = _os.path.join(str(_cfg.MMFB_ROOT), 'MMFakeBench_val.json')
OUT_DIR         = _os.path.join(str(_cfg.ROOT), 'experiments', 'overnight_v2')

os.makedirs(OUT_DIR, exist_ok=True)


def load_mmfb_val():
    img_raw = torch.load(VAL_IMG_PT, map_location='cpu').float()
    txt_raw = torch.load(VAL_TXT_PT, map_location='cpu').float()
    img     = F.normalize(img_raw, dim=-1)
    txt     = F.normalize(txt_raw, dim=-1)
    probs   = np.load(VAL_PROBS_NPY)
    sims    = np.load(VAL_SIMS_NPY)
    ids_df  = pd.read_csv(VAL_IDS_CSV)

    ateeq_df   = pd.read_csv(ATEEQ_CSV)
    deberta_df = pd.read_csv(DEBERTA_CSV)

    assert len(img) == len(ids_df) == len(ateeq_df) == len(deberta_df) == 1000, "Length mismatch!"

    labels    = (ids_df['gt_answers'] == 'Fake').astype(int).values
    fake_cls  = ids_df['fake_cls'].values
    categories = ['original', 'mismatch', 'textual_veracity_distortion', 'visual_veracity_distortion']

    ateeq_scores   = ateeq_df['ateeq_score_ft'].values
    deberta_scores = deberta_df['deberta_score'].values

    scalars = torch.tensor(
        np.stack([probs, sims, ateeq_scores], axis=1), dtype=torch.float32
    )

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


# ==== Task body ====

from sklearn.metrics import roc_auc_score
from itertools import product

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data   = load_mmfb_val()

with open(os.path.join(OUT_DIR, 'task1_baseline.json')) as f:
    t1 = json.load(f)
all_probs = np.array(t1['all_probs'])
labels    = data['labels']
fake_cls  = data['fake_cls']
categories = data['categories']

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
    print(f"  {cat}: thr={best_thr:.2f} -> {best_acc:.4f}")

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
