# Data & model weights (gitignored)

Large files are **not** tracked in git. A fresh clone has the code, `config.py`,
the tiny `results/*.json` evidence, and the two whitelisted inference artifacts
(`models/fusion_aitr/scalar_scaler.joblib`, `models/fusion_aitr/frozen_threshold.json`).
Everything else below must be obtained or regenerated locally.

## Configure paths
Everything resolves from one environment variable:

```
set PCL_ROOT=D:\Pics Can Lie          # repo root (default already points here)
set PCL_MMFB_ROOT=E:\Pics Can Lie\dataset\MMFakeBench   # MMFakeBench lives on E:\
set PCL_DATA_ROOT=%PCL_ROOT%\datasets # override if datasets live elsewhere
```
See `config.py` — every dataset / model / feature / cache / results path is derived
relative to `PCL_ROOT`.

## What lives where (all gitignored)
- `datasets/dataset/` — NewsCLIPpings (annotations, metadata, `origin/origin/` images, articles).
- `PCL_MMFB_ROOT` (E:\) — MMFakeBench (`MMFakeBench_val.json`, `MMFakeBench_val/` images).
- `models/clip_finetuned_v2/` — fine-tuned CLIP ViT-L/14 head (`clip_classifier.pt`) + `val_features/`.
- `models/fusion_aitr/aitr_weights.pt` — AITR fusion weights. (scaler + threshold ARE tracked.)
- `models/{blip_itm_finetuned,blip2_finetuned,ai_detector_finetuned}/` — other checkpoints.
- `features/*.csv`, `features/*.npy` — precomputed scores/features.
- `cache/` — API/page caches (SightEngine, Wikipedia, etc.).

## Regenerate the key artifacts
- Scaler + frozen threshold + honest VAL: `python scripts/fix_newsclip_inference.py`
- Honest TEST (GPU pass over test images): `python scripts/run_test_inference_fixed.py`
- DeBERTa v3 rescore (GPU): `python scripts/deberta_rescore_v3.py`
- MMFakeBench honest comparison + GBM: `python scripts/mmfakebench_honest.py`

## How to obtain the datasets
NewsCLIPpings and MMFakeBench are public research datasets — download from their
original sources and place under `datasets/dataset/` and `PCL_MMFB_ROOT` respectively.
Model weights are reproduced by the fine-tuning notebooks under `experiments/`.
