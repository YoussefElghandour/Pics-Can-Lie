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

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ AUDIT NOTE (Prompt B, Part 1) — THIS VETO IS AN ORACLE, NOT A RESULT.       ║
# ║ The condition below gates on `fake_cls == 'original'`, the GROUND-TRUTH     ║
# ║ MMFakeBench category, which does NOT exist at inference time. Every         ║
# ║ 'original' sample is REAL by construction, so the veto is trivially correct ║
# ║ there. It also reads `deberta` (article-NLI), not wiki_score, despite the   ║
# ║ "Wikipedia NLI veto" name. The deployable, category-agnostic version has    ║
# ║ precision ~0.23 and DROPS accuracy to ~0.537 (see veto_honest_report.json). ║
# ║ Retained ONLY for the ablation / negative-finding discussion. NEVER report  ║
# ║ the gated number (~0.798) as a headline — it is an upper bound that uses    ║
# ║ ground-truth metadata.                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
results = []
veto_thresholds    = np.arange(0.10, 0.91, 0.05)
uncertainty_bands  = [(0.0, 1.0), (0.3, 0.7), (0.2, 0.8), (0.35, 0.65)]

for deb_thr in veto_thresholds:
    for unc_lo, unc_hi in uncertainty_bands:
        veto_preds = base_preds.copy()
        orig_mask  = fake_cls == 'original'
        veto_cond = (
            orig_mask &
            (base_preds == 1) &
            (deberta >= deb_thr) &
            (all_probs >= unc_lo) & (all_probs <= unc_hi)
        )
        veto_preds[veto_cond] = 0
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
          f"overall={r['overall']:.4f} (delta{r['delta']:+.4f}) orig={r['orig_acc']:.4f}")

best_veto = results[0] if results else {}
if best_veto:
    print(f"\nBest veto: {best_veto}")
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
