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

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

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

all_probs = []
with torch.no_grad():
    for i in range(0, 1000, 64):
        logits = model(img[i:i+64], txt[i:i+64], scalars[i:i+64])
        all_probs.append(torch.sigmoid(logits).cpu())
all_probs = torch.cat(all_probs).numpy()

auc = roc_auc_score(labels, all_probs)
print(f"\nAUC-ROC: {auc:.4f}")

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

for cat in categories:
    best_cat = max(sweep, key=lambda x: x['per_cat'][cat])
    print(f"Best thr for {cat}: {best_cat['threshold']} -> {best_cat['per_cat'][cat]:.4f}")

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
