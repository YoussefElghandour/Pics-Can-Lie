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

scalars_B = data['scalars'].to(device)
scalars_C = torch.tensor(
    np.stack([data['probs'], data['sims'], data['deberta']], axis=1),
    dtype=torch.float32
).to(device)
scalars_D = torch.tensor(
    np.stack([data['probs'], data['sims'], data['ateeq'], data['deberta']], axis=1),
    dtype=torch.float32
).to(device)

TRAIN_DIR = _os.path.join(str(_cfg.ROOT), 'experiments', 'mmfakebench_training')
variant_paths = {
    'B': (os.path.join(TRAIN_DIR, 'aitr_mmfb_v2_B.pt'), 3, scalars_B),
    'C': (os.path.join(TRAIN_DIR, 'aitr_mmfb_v2_C.pt'), 3, scalars_C),
    'D': (os.path.join(TRAIN_DIR, 'aitr_mmfb_v2_D.pt'), 4, scalars_D),
}

variant_probs = {}
for vname, (vpath, n_scalars, vscalars) in variant_paths.items():
    if not os.path.exists(vpath):
        print(f"Variant {vname} checkpoint not found at {vpath} -- skipping")
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
