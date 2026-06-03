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

df  = pd.read_csv(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'VERITE.csv'))
print(f"CSV rows: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

img = np.load(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'VERITE_clip_image_embeddings_ViTL14.npy'))
txt = np.load(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'VERITE_clip_text_embeddings_ViTL14.npy'))
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
