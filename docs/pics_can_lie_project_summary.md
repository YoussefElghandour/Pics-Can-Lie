# Pics Can Lie — Full Project Summary
> Multimodal Misinformation Detection — Thesis Project
> Last updated: May 8, 2026

---

## 1. Project Overview

**Goal:** Detect out-of-context news image-caption pairs using a multi-signal fusion pipeline.

**Task:** Binary classification — REAL (image matches caption) vs FAKE (image is out-of-context).

**Primary dataset:** NewsCLIPpings Merged/Balanced — 71k train, 5k val, binary labels.

**Secondary dataset:** MMFakeBench — for AI-generated image detection evaluation.

**Ultimate target:** Beat MUSE-MLP (90.0%) and ideally MUSE AITR (93.3%) on NewsCLIPpings.

---

## 2. Hardware Setup — Two Machines

This project runs across TWO machines depending on the task. This is critical context.

### Machine A — Personal laptop/desktop
- **Drive:** `D:\Pics Can Lie\`
- **GPU:** Weaker GPU (used for light inference tasks like DeBERTa scoring)
- **Used for:** DeBERTa NLI scoring, Wikipedia fetching, data preparation, evidence link analysis
- **Key files here:**
  - `D:\Pics Can Lie\links_val.json` — evidence links from Abdelnabi et al.
  - `D:\Pics Can Lie\deberta_val_scores_v2.csv` — DeBERTa scores for v2 val samples
  - `E:\Pics Can Lie\wikipedia_text_cache.json` — cached Wikipedia API responses (3,243 entries, 966 filled)
  - `D:\Pics Can Lie\val_sample_ids.csv` — WARNING: this is OLD 3,232-sample file, NOT v2
  - `D:\Pics Can Lie\dataset\data\NewsClipPings\` — NewsCLIPpings dataset copy

### Machine B — 4090 workstation
- **Drive:** `K:\Joee El Ghandour\Pics Can Lie\`
- **GPU:** NVIDIA RTX 4090 (22.5GB VRAM)
- **Used for:** All heavy training (BLIP, CLIP), CLIP feature extraction, evidence scoring
- **All major model files and val features are here**

**IMPORTANT:** When a notebook says `K:\` it runs on the 4090. When it says `D:\` it runs on personal machine. Do NOT mix paths.

---

## 3. Folder Structure (4090 Machine — K:\)

```
K:\Joee El Ghandour\Pics Can Lie\
├── kaggle_dataset_full\
│   ├── images\origin\         ← news images (bbc, guardian, usa_today, washington_post)
│   ├── merged_balanced\       ← train.json, val.json, test.json (annotations)
│   └── metadata\              ← train.json, val.json, test.json
│       NOTE: article .txt files DO NOT EXIST locally — only JSON + images downloaded
├── dataset\MMFakeBench\       ← AI-gen detection dataset (Ateeq model)
├── models\
│   ├── blip-itm-large-coco\   ← BLIP large weights (~1.8GB)
│   └── clip\                  ← CLIP ViT-L/14 weights (~890MB)
├── blip_itm_large_finetuned\  ← BLIP v4 best checkpoint (78%)
│   ├── model.safetensors
│   ├── val_cls_embeddings_with_ids.pt    ← (5000, 1024) — NOT discriminative, don't use
│   ├── val_clip_similarities_with_ids.npy ← (5000,) frozen CLIP sims
│   ├── val_labels_with_ids.npy
│   └── val_sample_ids.csv                ← (5000 rows)
├── clip_finetuned\            ← CLIP v1 best checkpoint (83.58%)
│   └── val_features\
│       ├── clip_finetuned_probs.npy      ← (5000,)
│       ├── clip_finetuned_sims.npy       ← (5000,)
│       ├── clip_img_features.pt          ← (5000, 768)
│       ├── clip_txt_features.pt          ← (5000, 768)
│       └── val_sample_ids.csv            ← (5000 rows)
├── clip_finetuned_v2\         ← CLIP v2 best checkpoint (85.6%) ← CURRENT BEST
│   └── val_features\
│       ├── clip_finetuned_probs.npy      ← (5000,) ← USE THIS for fusion
│       ├── clip_finetuned_sims.npy       ← (5000,) ← USE THIS for fusion
│       ├── clip_img_features.pt          ← (5000, 768)
│       ├── clip_txt_features.pt          ← (5000, 768)
│       └── val_sample_ids.csv            ← (5000 rows) ← MASTER ID FILE
├── deberta_val_scores_v2.csv  ← DeBERTa scores aligned to v2 CLIP IDs ← USE THIS
├── links_val.json             ← needs copying from D:\ (25.6MB)
├── evidence_clip_scores.csv   ← NOT YET COMPUTED
├── fusion_evidence\           ← NOT YET CREATED
├── clip_finetune_v2.ipynb     ← CLIP v2 notebook (current best)
├── evidence_pipeline.ipynb    ← NEW — not yet run
└── fusion_mlp.ipynb           ← basic fusion — not yet run with v2 signals
```

---

## 4. All Models Tried — Results

### 4.1 BLIP ITM Base v1-v2
- **Config:** 2-4 unfrozen layers, LR 5e-6, batch 16, 50k-71k samples
- **Result:** Val F1 = 0.7511-0.7636, Val Acc = 75.11-76.40%
- **Status:** Superseded

### 4.2 BLIP ITM Large v4 (Kaggle → 4090)
- **Config:** Vision layers 18-23 + text 6-11 unfrozen, LR 3e-6, batch 32, patience 5
- **Result:** Val F1 = 0.7794, Val Acc = 78.00% at epoch 10/15
- **Training time:** ~5 hrs/epoch Kaggle T4, ~17 min/epoch 4090
- **Status:** Done. Saved to `blip_itm_large_finetuned\`
- **Critical:** CLS embeddings (1024-dim) NOT discriminative (F1=0.46). Only ITM scalar is useful.

### 4.3 Fine-tuned CLIP ViT-L/14 v1 (4090)
- **Config:** Vision+text blocks 20-23 (4 each), LR head=1e-6, encoder=1e-7, batch 32, dropout=0.3
- **Head:** [img_feat(768) | txt_feat(768) | cosine_sim(1)] → 512 → 128 → 1
- **Result:** Val F1 = 0.8356, Val Acc = 83.58% at epoch 10/15
- **Training time:** ~257 minutes total on 4090
- **NaN issue fix:** `clip_model.float()` + freeze projection layers + LR 1e-6

### 4.4 Fine-tuned CLIP ViT-L/14 v2 (4090) ← CURRENT BEST
- **Config:** Vision+text blocks 20-23 (4 each), LR head=5e-5, encoder=5e-7 (÷100), batch 64, dropout=0.5, weight_decay=0.05, BatchNorm in head, num_workers=2
- **Key changes from v1:** Higher head LR, much slower encoder LR (÷100 vs ÷10), stronger regularization, BatchNorm, projection layers frozen
- **Result:** Val F1 = 0.852, Val Acc = 85.6%
- **Training curve:**
  ```
  Epoch 1: F1=0.7509, Acc=77.14%
  Epoch 2: F1=0.8215, Acc=83.10%
  Epoch 3: F1=0.8285, Acc=83.90%
  Epoch 4: F1=0.8400, Acc=84.62%
  Epoch 5: F1=0.8478, Acc=85.26% ← best (confirmed 85.6% final)
  ```
- **Signal diagnostics:**
  - CLIP probs alone: F1=0.8520
  - CLIP sims alone: F1=0.8312
  - DeBERTa NLI alone: F1=0.4638 (weak)
  - All 3 (LR): estimated ~85-87%

---

## 5. DeBERTa NLI Scoring — Status

### Current version (v2) ← USE THIS
- **File:** `deberta_val_scores_v2.csv` on both machines
- **Scores:** min=0.000, max=1.000, mean=0.418
- **Coverage:** 5000/5000 matched to v2 CLIP IDs (100% alignment)
- **Method:** Wikipedia REST API via `caption_entities_rel` top-2 entities by confidence
- **Run on:** Personal machine (D:\) using weaker GPU
- **Wikipedia cache:** `E:\Pics Can Lie\wikipedia_text_cache.json` (3,243 entries, 966 filled)

### Why article text was not used
Article .txt files DO NOT exist in local dataset. Only images + JSON downloaded. Use Wikipedia API instead.

### Old broken version — DO NOT USE
- **File:** `deberta_val_scores.csv` (no v2 suffix)
- **Issue:** Scores all near 0.014 — from 300-sample old experiment with broken path resolver

---

## 6. Fusion — What Was Tried and What's Next

### 6.1 Old fusion — OBSOLETE
BLIP ITM + DeBERTa + SightEngine API → logistic regression on 300 samples. SightEngine removed (cost).

### 6.2 BLIP CLS embeddings fusion — FAILED
- Result: 70.90% — WORSE than BLIP alone (78%)
- Why: CLS embeddings not discriminative (F1=0.46). ITM head is what makes BLIP useful.

### 6.3 Basic 3-signal fusion — NOT YET RUN WITH V2
- Signals: fine-tuned CLIP prob + CLIP sim + DeBERTa (3 scalars)
- Expected: ~85-87%
- Decision: Skipped in favor of evidence pipeline which gives bigger gains

### 6.4 Evidence pipeline fusion — NOT YET RUN ← NEXT STEP
- 8 signals: clip_prob, clip_sim, deberta, s2, s3, s4, s5, s6
- s2-s6 from downloading evidence images and computing CLIP similarities
- Expected: 88-93% — targeting above MUSE (90%)
- Notebook: `evidence_pipeline.ipynb`

---

## 7. Evidence Retrieval — Full Status

### What we have
- `links_val.json` (25.6MB) from Abdelnabi et al. Google Drive (on personal machine D:\)
- 6,972 val samples with evidence links
- Coverage of our 5,000 val samples: 99.3% after ID mapping
- Link liveness (May 2026): 70% alive, 27% dead, 3% timeout

### Critical: ID mapping required
```python
# links_val.json uses integer index keys (0,1,2...) NOT article IDs
idx_to_id   = {str(i): str(a["id"]) for i, a in enumerate(annotations)}
id_to_links = {aid: links_data[idx] for idx, aid in idx_to_id.items() if idx in links_data}
```

### The 6 evidence CLIP scores
```
s2 = clip_sim(original_image, direct_search_image)
s3 = clip_sim(caption,        direct_search_image)
s4 = clip_sim(original_image, inverse_search_image)
s5 = clip_sim(caption,        inverse_search_image)
s6 = clip_sim(direct_image,   inverse_image)  ← cross-evidence
```

### Why this should beat MUSE
- MUSE: frozen CLIP (80.7%) + evidence → 90.0% (+9.3%)
- Ours: fine-tuned CLIP (85.6%) + same evidence → estimated 88-93%

### Sahar Abdelnabi contact
- **Email:** sahar.abdelnabi@cispa.de
- **Requested:** Pre-downloaded evidence images (same ones used in MUSE experiments)
- **Reason:** 30% dead links since 2022 affects reproducibility
- **Status:** Two emails sent, no response yet as of May 8, 2026
- **If she responds:** Use her pre-downloaded images instead of re-crawling for better reproducibility

---

## 8. What We Did NOT Do and Why

### BLIP-2 / ViT-G
- Cannot fit on single 4090 (needs 35-40GB). Expected gain only +1-3%. Not worth it.

### ViT-L/14@336px (higher resolution)
- 2x VRAM at batch 32 — pushes past 22.5GB. Marginal gain for news photos (semantic not resolution is bottleneck).

### Full CLIP unfreezing
- 119.5M params already caused overfitting on 71k. Full model would be worse.

### Cross-attention fusion
- Engineering complexity for marginal gain. Simple MLP on scalars sufficient and more interpretable.

### Text evidence scraping (for s4/s5 text-based)
- Requires scraping HTML, parsing, rate limiting — complex. Image evidence alone (s2,s3,s6) gives most of the gain. Can add later.

### Ateeq on NewsCLIPpings
- NewsCLIPpings = real news photos, not AI-generated. Ateeq outputs near-constant "not AI" = zero variance = noise.
- Ateeq reserved for MMFakeBench only.

### MMFakeBench evaluation
- Focused on NewsCLIPpings first. Planned after evidence pipeline complete.

### Claude API explanation
- Accuracy first, explainability last. Planned as final component.

### Basic 3-signal fusion (CLIP + DeBERTa MLP)
- DeBERTa F1=0.4638 — too weak for meaningful gain (+1-2%). Jumped straight to evidence pipeline (+5-10%).

---

## 9. Specific Failures and Fixes

| Failure | Cause | Fix |
|---|---|---|
| CLIP NaN loss | fp16 overflow in backprop | `clip_model.float()` + freeze projection layers |
| Label alignment 49.5% | Different shuffle order between runs | Re-extract with IDs via `blip_reinference.ipynb` |
| DeBERTa scores all 0.014 | Old 300-sample cache loaded | Delete cache, recompute as `deberta_val_scores_v2.csv` |
| Article files missing | .txt files not in local download | Use Wikipedia API via `caption_entities_rel` |
| Wikipedia cache 0 entries | Wrong path (D:\ vs E:\) | Set `WIKI_CACHE = r"E:\Pics Can Lie\wikipedia_text_cache.json"` |
| MUSE 93.3% misunderstood | Thought it was model accuracy | It's system accuracy. Their frozen CLIP alone = 80.7% |
| v1/v2 CLIP ID mismatch | Different random shuffles | Re-ran DeBERTa scoring with v2 ID file as master |
| links_val.json 0% overlap | Index keys vs article IDs | Map through annotations: `idx_to_id = {str(i): str(a["id"])...}` |
| Fusion worse than baseline | BLIP CLS embeddings (1024-dim noise) | Use ITM scalar or fine-tuned CLIP probability |

---

## 10. Key Results Summary

| Model/System | Acc | F1 | Notes |
|---|---|---|---|
| CLIP RN101 (paper baseline) | 72.44% | — | Literature |
| VERITE CLIP ViT-L/14 | 74.40% | — | Literature |
| BLIP ITM base v2 | 75.11% | 0.7511 | Our result |
| BLIP ITM base v3 | 76.40% | 0.7636 | Our result |
| BLIP ITM large v4 | 78.00% | 0.7794 | Our result |
| Frozen CLIP ViT-L/14 | ~79.33% | 0.7933 | Our diagnostic |
| Fine-tuned CLIP v1 | 83.58% | 0.8356 | Our result |
| COSMOS | 85.00% | — | Literature |
| **Fine-tuned CLIP v2** | **85.60%** | **0.852** | **Current best** |
| Fusion MLP (broken CLS) | 70.90% | 0.7127 | Failed |
| SNIFFER | 88.40% | — | Literature, needs Google API |
| MUSE-MLP | 90.00% | — | Literature, needs Google API evidence |
| MUSE AITR | 93.30% | — | Literature, needs Google API evidence |
| **Evidence fusion (planned)** | **88-93%** | **TBD** | Next step |

---

## 11. Important Code Notes

### Path resolver (images)
```python
def resolve_image_path(rel_path):
    return os.path.join(DATASET_ROOT,
        str(rel_path).replace("visual_news/", "images/"))
```

### Master ID file (always use this)
```python
# 5000 rows — v2 CLIP val samples
MASTER_IDS = r"K:\Joee El Ghandour\Pics Can Lie\clip_finetuned_v2\val_features\val_sample_ids.csv"
# D:\Pics Can Lie\val_sample_ids.csv has only 3232 rows — DO NOT USE
```

### CLIP loading (mandatory steps)
```python
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
clip_model = clip_model.float()   # REQUIRED — prevents NaN from fp16
# Do NOT unfreeze: clip_model.visual.proj or clip_model.text_projection
```

### CLIP layer counts
```python
# CLIP ViT-L/14: 24 vision blocks (0-23), 24 text blocks (0-23)
# Unfreeze last 4: start=20 for both
# BLIP ITM large: 24 vision (0-23), 12 text (0-11) — asymmetric!
```

### DeBERTa output order
```python
# cross-encoder/nli-deberta-v3-large:
# probs[0]=contradiction, probs[1]=neutral, probs[2]=entailment ← use this
```

### Evidence ID mapping
```python
idx_to_id   = {str(i): str(a["id"]) for i, a in enumerate(annotations)}
id_to_links = {aid: links_data[idx] for idx, aid in idx_to_id.items() if idx in links_data}
```

### Correct fusion feature matrix (v2)
```python
# 3-signal basic:
X = np.stack([clip_probs, clip_sims, deb_scores], axis=1)  # (5000, 3)

# 8-signal evidence:
X = np.stack([clip_probs, clip_sims, deb_scores, s2, s3, s4, s5, s6], axis=1)  # (5000, 8)

# WRONG — never do this:
X = np.concatenate([blip_cls_emb, clip_sims, deb_scores], axis=1)  # (5000, 1026) = noise
```

---

## 12. Notebooks

| Notebook | Machine | Purpose | Status |
|---|---|---|---|
| `downloadmodels.ipynb` | Personal | Download BLIP + CLIP locally | ✅ Done |
| `blip_itm_large_enhanced.ipynb` | 4090 | Fine-tune BLIP ITM large | ✅ Done (78%) |
| `blip_reinference.ipynb` | 4090 | Re-extract BLIP embeddings with IDs | ✅ Done |
| `wikipedia_deberta_scoring.ipynb` | Personal | DeBERTa NLI via Wikipedia | ✅ Done (v2) |
| `clip_finetune.ipynb` | 4090 | CLIP v1 fine-tuning | ✅ Done (83.58%) |
| `clip_finetune_v2.ipynb` | 4090 | CLIP v2 fine-tuning (improved) | ✅ Done (85.6%) |
| `fusion_mlp.ipynb` | Either | Basic 3-signal MLP | ❌ Not run with v2 |
| `evidence_pipeline.ipynb` | 4090 | Full evidence crawl + score + fuse | ❌ Not yet run |
| `finetune_ai_detector.py` | — | Ateeq AI-gen detector | ⚠️ Trained, not integrated |

---

## 13. What NOT to Do

1. **Do not use `D:\Pics Can Lie\val_sample_ids.csv`** — 3,232 rows (old). Use v2 file (5,000 rows).
2. **Do not use `deberta_val_scores.csv`** — broken. Use `deberta_val_scores_v2.csv`.
3. **Do not use BLIP CLS embeddings** — F1=0.46, noise in fusion.
4. **Do not unfreeze CLIP projection layers** — NaN loss.
5. **Do not load CLIP without `.float()`** — fp16 overflow.
6. **Do not align features by order** — always align by ID.
7. **Do not compare model accuracy to MUSE system accuracy** — apples vs oranges.
8. **Do not use article text** — files don't exist locally.
9. **Do not add Ateeq to NewsCLIPpings** — real photos, Ateeq = noise there.
10. **Do not use index keys from links_val.json directly** — must map through annotations.

---

## 14. Next Steps — Priority Order

### Immediate (before next session)
- [ ] Copy `links_val.json` from `D:\` to `K:\Joee El Ghandour\Pics Can Lie\`
- [ ] Copy `deberta_val_scores_v2.csv` from `D:\` to `K:\Joee El Ghandour\Pics Can Lie\`

### Step 1 — Evidence pipeline (highest priority)
- Run `evidence_pipeline.ipynb` on 4090
- Cell 5 is slow (2-4 hours, network dependent)
- Saves every 100 samples — safe to interrupt and resume
- Alternative: pre-download images on personal machine, modify Cell 5 to load from disk

### Step 2 — Wait for Sahar's response
- If she sends pre-downloaded evidence: use those instead of crawled images
- Same data MUSE used = more reliable comparison with their reported numbers

### Step 3 — Basic fusion (optional, may skip)
- Run `fusion_mlp.ipynb` with v2 features and `deberta_val_scores_v2.csv`
- Expected ~85-87% — skip if evidence pipeline already exceeds this

### Step 4 — MMFakeBench evaluation
```python
X = np.stack([clip_probs, clip_sims, deb_scores, ateeq_scores], axis=1)
```

### Step 5 — Update thesis documentation
- Fill TBD values with evidence pipeline results
- Update comparison table

### Step 6 — Claude API explanation generation
- Final pipeline component after accuracy is finalized
