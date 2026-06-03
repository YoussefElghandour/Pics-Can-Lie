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
