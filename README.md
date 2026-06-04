# Pics Can Lie — Multimodal Misinformation Detection

Out-of-context image detection system for news misinformation.
The core idea: a real news image paired with a mismatched caption is "out-of-context"
misinformation — the image itself is authentic but is being used deceptively.

**Primary dataset:** NewsClipPings (85k samples, BBC / Guardian / USA Today / Washington Post)
**Secondary dataset:** MMFakeBench (11k samples, mixed manipulation types)
**Stack:** Python 3.10, PyTorch, HuggingFace Transformers, Gradio, Ollama

---

## Results

The system detects out-of-context news images using a fine-tuned **CLIP ViT-L/14 → AITR
transformer fusion**, with an isotonic-calibrated decision. Performance is reported with **AUC
as the primary metric** (threshold-free) and accuracy at a calibrated operating point.

**NewsCLIPpings test set (7264 samples, balanced real/out-of-context):**

| Metric | Value |
|---|---|
| **AUC** | **0.932** [0.926, 0.937] |
| **Accuracy** | **0.865** [0.856, 0.873] (calibrated decision @0.5) |
| Per-source accuracy | BBC 0.845 · Guardian 0.863 · USA Today 0.877 · Washington Post 0.870 |

The decision uses a calibrated probability (isotonic), which also serves as a well-calibrated
confidence score (expected calibration error 0.017). The system uses **no external APIs or
evidence retrieval at inference** — it is fully self-contained, in contrast to evidence-based
systems such as SNIFFER (88.4%, external entity/LLM retrieval) or MUSE (90–93%, Google API
evidence). On the out-of-domain MMFakeBench benchmark, generalization is limited (best
deployable AUC ≈ 0.79), reflecting that the in-domain consistency signal does not fully transfer
to other manipulation types.

Full numbers, confidence intervals, and ablations: **`docs/corrected_results.md`** and
**`docs/comparison_table.md`**.

---

## Live demo (`src/app.py`)

Two stages, three independent signals; the verdict is anchored only to consistency.

| Signal | Model | Role |
|---|---|---|
| **Image–caption consistency** | CLIP ViT-L/14 → AITR → isotonic calibration | **primary** — drives REAL/FAKE (out-of-context) |
| **Image origin** | Ateeq (fine-tuned Siglip) AI-vs-real detector | auxiliary — "is the image AI-generated?" for wild uploads |
| **Caption factuality** | Claude Sonnet 4.6 + web search (on-demand) | auxiliary — verifies the caption's factual claims |

Ateeq was validated on a wild-image test to genuinely generalize to unseen generators (StyleGAN,
Midjourney), with a known resolution bias on large real photos. The demo is fully local except the
optional Claude fact-check. *(SightEngine and DeBERTa/Wikipedia NLI were retired.)*

---

## Datasets

### NewsClipPings (primary)
- ~85,000 samples (71k train / 7k val / 7k test), balanced real vs. out-of-context
- Images are always authentic and unedited — misinformation is purely from image-caption mismatch
- Captions are factually true, so fact-checking modules are uninformative on this dataset
- `dataset/data/NewsClipPings/merged_balanced/` — labels; `metadata/` — captions + image paths
- `dataset/origin/origin/` — raw images (bbc, guardian, usa_today, washington_post)

### MMFakeBench (secondary)
- ~11,000 samples (6k train / 1k val / 10k test), mixed manipulation types
- 30% text-manipulated, 10% PS-edited images, 10% AI-generated images, 30% out-of-context
- Captions can contain false claims → fact-checking and AI detection are meaningful here
- `dataset/MMFakeBench/`

---

## Modules Implemented

### 1. BLIP ITM — Image-Text Matching
**File:** `finetune_blip_itm.ipynb`  
**Checkpoint:** `blip_itm_finetuned/`

- Model: `Salesforce/blip-itm-base-coco`
- Fine-tuned on 50,000 balanced samples (25k real + 25k fake)
- Task: binary classification — does the image match the caption?
- Training: 9 epochs, LR 1e-5, batch 16, early stopping (patience 3)
- **Result: 74.9% accuracy, 75.1% F1** (epoch 9) — +13 pp over pretrained zero-shot (~62%)

| Epoch | Train Loss | Val F1 | Val Acc |
|---|---|---|---|
| 1 | 0.655 | 72.3% | 73.0% |
| 5 | 0.470 | 73.5% | 74.5% |
| 9 | 0.437 | **75.1%** | **74.9%** |

---

### 2. DeBERTa NLI — Article-Caption Consistency
**File:** `text_nli_deberta.ipynb`

- Model: `cross-encoder/nli-deberta-v3-large`
- Extracts the 3 most relevant sentences from the article via TF-IDF cosine similarity
- Runs NLI entailment: does the article text support the caption?
- Returns an entailment probability score per sample

**Standalone evaluation (NewsClipPings, n=1000, threshold 0.65):**
- Accuracy: 60.2%, F1: 70.6% (inflated — strong class bias)
- Fake recall: 95.8% (479/500) — but only 24.6% real class accuracy (377/500 real misclassified as fake)
- Threshold sweep 0.3→0.7 produces only 57.6%–60.2% — no meaningful decision boundary
- As a fusion feature: contributes +2.3 pp over BLIP alone

**Standalone evaluation script:** `deberta_standalone_eval.py` → results in `deberta_standalone_eval.json`

---

### 3. AI Image Detector — SightEngine API (replaced)
**File:** `image_authenticity_sightengine.ipynb`

- Uses the SightEngine `genai` API to detect AI-generated images
- Tested on 10 real news photos + 4 AI-generated images (~100% on small test set, not reliable)
- Results cached in `sightengine_cache.json`
- **Status: replaced by fine-tuned local model in next fusion retrain**

---

### 4. AI Image Detector — Hive API (replaced)
**File:** `image_authenticity_hive.ipynb`

- Uses the Hive AI-Generated Image Detection API
- Returns score 0 (real) to 1 (AI-generated)
- Results cached in `hive_cache.json`
- **Status: not used in fusion — SightEngine was used instead; both now superseded by fine-tuned detector**

---

### 5. AI Image Detector — Fine-Tuned (Ateeq SiglipForImageClassification)
**File:** `finetune_ai_detector.py`  
**Checkpoint:** `ai_detector_finetuned/`

The pretrained `Ateeqq/ai-vs-human-image-detector` model was evaluated and found to fail badly
on news-domain images before fine-tuning:

| Metric | Before | After | Delta |
|---|---|---|---|
| Overall accuracy | 56.0% | 75.5% | +19.5 pp |
| Overall F1 | 52.2% | 78.0% | +25.9 pp |
| FP rate on VisualNews real photos | 47.9% | 29.6% | −18.3 pp |
| PS-edited image detection | 24.0% | 84.0% | +60.0 pp |
| AI-generated image detection | 72.0% | 90.0% | +18.0 pp |

**Training setup:**
- 500 real NewsClipPings images (seed=42) + 500 MMFakeBench samples (250 AI-generated + 250 PS-edited)
- 800 train / 200 val, stratified — 6 epochs, 2-phase (classifier-only → + last 2 encoder layers)
- ~2 minutes on CUDA

**Generalization verified on 3 independent test sets (FP rate on real images):**

| Test Set | Before | After |
|---|---|---|
| MMFakeBench val VisualNews (held-out) | 47.9% | 29.6% |
| New NewsClipPings sample (seed=99, 200 imgs) | 67.0% | 27.0% |
| NewsClipPings val.json real (100 imgs) | 68.0% | 16.0% |

Val loss fell every epoch with no train/val divergence — clean convergence confirmed.

**5-fold CV standalone on MMFakeBench:**

| Model | Accuracy | F1 | Std |
|---|---|---|---|
| Pretrained | 56.5% | 52.7% | ±9.6% |
| Fine-tuned | **80.0%** | **80.1%** | ±5.0% |

Scores cached in `mmfakebench_ai_scores.csv` (pretrained) and `mmfakebench_ai_scores_finetuned.csv` (fine-tuned).

---

### 6. Wikipedia Fact-Check Module
**Files:** `wikipedia_factcheck.py`, `wikipedia_factcheck_newsclip_spotcheck.py`  
**Cached scores:** `mmfakebench_factcheck_scores.csv`, `mmfakebench_factcheck_checkpoint.csv`, `newsclip_factcheck_spotcheck.csv`

- Pipeline: spaCy NER → Wikipedia API → DeBERTa NLI entailment between Wikipedia text and caption
- Returns a `factcheck_score` (entailment probability) and `factcheck_available` binary flag

**Evaluation results:**

| Metric | MMFakeBench (1000 samples) | NewsClipPings (200 samples) |
|---|---|---|
| Coverage (status: ok) | 54.0% | 92.0% |
| No Wikipedia result | 46.0% | 8.0% |
| Real mean factcheck_score | +0.0053 | −0.0583 |
| Fake mean factcheck_score | −0.0200 | −0.0708 |
| Separation (Δmean) | 0.0147 | 0.0125 |
| Threshold accuracy (< 0.0 → fake) | 33.4% | 49.5% |

**Conclusion:** Signal separation is negligible on both datasets. Out-of-context misinformation uses
factually correct captions — Wikipedia NLI cannot distinguish real from fake.
Module retained in the live pipeline for textual fabrication claims only.
**Excluded from fusion training.**

---

### 7. Fusion Classifiers — Logistic Regression (5-fold CV)
All fusion experiments use `sklearn` Logistic Regression with 5-fold cross-validation.

#### 7a. Pretrained BLIP + DeBERTa + SightEngine
**File:** `fusion_classifier.ipynb` → `fusion_classifier_results.png`
- 1000 samples, features in `fusion_features_1000.csv` / `fusion_features_1000_3feat.csv`

#### 7b. Fine-Tuned BLIP + DeBERTa
**File:** `fusion_finetuned_blip.ipynb`
- 1000 samples, features in `fusion_features_finetuned_1000.csv`

#### 7c. Fine-Tuned BLIP + DeBERTa + SightEngine — full ablation
**File:** `fusion_3module_finetuned.ipynb` → `fusion_3module_finetuned_results.png`  
**Script:** `fusion_ablation.py` → `ablation_results.json`

All 4 configurations evaluated on the **same 300 samples** (fair comparison):

| Configuration | Accuracy | F1 | vs BLIP only |
|---|---|---|---|
| BLIP only | 82.67% | 82.56% | — |
| BLIP + DeBERTa | 85.0% | 84.84% | +2.3 pp |
| BLIP + SightEngine | 88.0% | 88.09% | +5.4 pp |
| **BLIP + DeBERTa + SightEngine** | **89.67%** | **89.58%** | **+7.0 pp** |

Trained fusion model saved as `fusion_model.joblib` + `fusion_scaler.joblib`.  
Additional feature cache: `fusion_features.csv`, `fusion_features_2000.csv`, `fusion_features_300_3feat_finetuned.csv`.

---

### 8. Gradio Demo App
**File:** `app.py`

- Full interactive web UI using Gradio
- Modules: fine-tuned BLIP ITM + DeBERTa NLI + SightEngine AI detection
- User uploads image + caption → prediction (real / out-of-context) + per-module score breakdown
- **Pending:** replace SightEngine with fine-tuned local AI detector; fix LLaMA VRAM issue

---

### 9. Batch Evaluation Scripts
- `evaluate.py` — runs full pipeline on 100 balanced test samples (50 real + 50 fake), prints accuracy / precision / recall / F1
- `mmfakebench_eval.py` — MMFakeBench module evaluation script
- `deberta_standalone_eval.py` — DeBERTa standalone evaluation with threshold sweep
- `verify_overfitting.py` — FP rate verification across 3 independent test sets for the fine-tuned AI detector
- `test_model.py` — quick model loading sanity check

---

### 10. BLIP-2 Pipeline (blocked by hardware)
**Files:** `blip2_full_pipeline.ipynb`, `train_blip2_local.py`, `train_blip2_local.ipynb`, `evaluate_blip2_pretrained.ipynb`, `evaluate_blip2_pretrained.py`  
**Checkpoint (partial):** `blip2_finetuned/`

- Model: `Salesforce/blip2-opt-2.7b`
- Strategy: freeze all → unfreeze Q-Former + language_projection + last 2 vision encoder layers
- Task framed as text generation: prompt → "real" or "out-of-context"
- Zero-shot eval implemented and working
- Fine-tuning loop implemented with fp16, GradScaler, gradient accumulation (effective batch 16)
- **Status: blocked** — local machine insufficient RAM/pagefile for the 5.4 GB model; Kaggle GPU required
- Export utilities: `kaggle_export_minimal.py` (5.5k images, ~540 MB), `kaggle_export_full.py` (71k images, ~2.7 GB)
- Pre-built zip archives: `kaggle_dataset_minimal.zip`, `kaggle_dataset_full.zip`

---

## What Was Not Completed

| Item | Reason |
|---|---|
| BLIP-2 fine-tuning | Local hardware insufficient; requires Kaggle GPU |
| Fusion retrain with fine-tuned AI detector | Pending — scores cached, retrain not yet run |
| Fusion retrain with Wikipedia feature | Pending — module excluded from training (negligible signal) |
| Full fusion on MMFakeBench | Not necessary — module-level evaluation is sufficient for thesis |
| DeBERTa at scale (>1000 samples) | Class bias makes standalone metric uninformative beyond current eval |
| Hive API at scale | API cost limits; superseded by local fine-tuned detector |
| LLaMA 3.1 8B explanation layer | VRAM issues not resolved; not integrated into final pipeline |

---

## Next Fusion Configuration (pending retrain)

| Feature | Source | Status |
|---|---|---|
| `blip_score` | BLIP ITM fine-tuned | ✅ active |
| `deberta_score` | DeBERTa NLI | ✅ active |
| `ai_detection_score` | Fine-tuned Ateeq detector (replaces SightEngine) | ⏳ pending retrain |
| `factcheck_score` | Wikipedia NLI (0.0 when missing) | ⏳ pending retrain |
| `factcheck_available` | Binary Wikipedia coverage flag | ⏳ pending retrain |

---

## File Map

```
Pics Can Lie/
├── app.py                               Gradio demo (BLIP ITM + DeBERTa + SightEngine)
├── evaluate.py                          Batch eval — 100 test samples
├── mmfakebench_eval.py                  MMFakeBench module evaluation
├── deberta_standalone_eval.py           DeBERTa standalone eval + threshold sweep
├── fusion_ablation.py                   Fusion ablation script (4 configurations)
├── finetune_ai_detector.py              AI detector fine-tuning script
├── wikipedia_factcheck.py               Wikipedia NLI fact-check module (MMFakeBench)
├── wikipedia_factcheck_newsclip_spotcheck.py  Wikipedia spot-check on NewsClipPings
├── verify_overfitting.py                FP rate generalization verification
├── test_model.py                        Model loading sanity check
├── kaggle_export_minimal.py             Export 5.5k images for Kaggle
├── kaggle_export_full.py                Export 71k images for Kaggle
│
├── finetune_blip_itm.ipynb              BLIP ITM fine-tuning (74.9% / F1 75.1%)
├── text_nli_deberta.ipynb               DeBERTa NLI module
├── image_authenticity_sightengine.ipynb SightEngine AI detection
├── image_authenticity_hive.ipynb        Hive AI detection
├── fusion_classifier.ipynb              Fusion v1 (pretrained BLIP + DeBERTa + SightEngine)
├── fusion_finetuned_blip.ipynb          Fusion v2 (fine-tuned BLIP + DeBERTa)
├── fusion_3module_finetuned.ipynb       Fusion v3 ablation (best: 89.67%)
├── blip2_full_pipeline.ipynb            BLIP-2 zero-shot eval + fine-tuning (Kaggle-ready)
├── train_blip2_local.ipynb              BLIP-2 training (local, blocked by hardware)
├── evaluate_blip2_pretrained.ipynb      BLIP-2 pretrained evaluation
├── dataset_exploration.ipynb            Dataset structure analysis
│
├── blip_itm_finetuned/                  BLIP ITM checkpoint (epoch 9)
├── ai_detector_finetuned/               Fine-tuned Ateeq SiglipForImageClassification
├── blip2_finetuned/                     BLIP-2 partial checkpoint
├── fusion_model.joblib                  Trained fusion classifier
├── fusion_scaler.joblib                 Fusion feature scaler
│
├── fusion_features.csv                  Feature cache (general)
├── fusion_features_1000.csv             Pretrained BLIP + DeBERTa (1000 samples)
├── fusion_features_1000_3feat.csv       Pretrained 3-module (1000 samples)
├── fusion_features_2000.csv             Extended feature cache
├── fusion_features_finetuned_1000.csv   Fine-tuned BLIP + DeBERTa (1000 samples)
├── fusion_features_300_3feat_finetuned.csv  Fine-tuned 3-module (300 samples, ablation)
├── mmfakebench_ai_scores.csv            Pretrained AI detector scores (MMFakeBench)
├── mmfakebench_ai_scores_finetuned.csv  Fine-tuned AI detector scores (MMFakeBench)
├── mmfakebench_factcheck_scores.csv     Wikipedia fact-check scores (MMFakeBench, 1000)
├── mmfakebench_factcheck_checkpoint.csv Wikipedia fact-check checkpoint (partial)
├── newsclip_factcheck_spotcheck.csv     Wikipedia fact-check spot-check (NewsClipPings, 200)
├── mmfakebench_fusion_results.json      MMFakeBench fusion evaluation results
├── ablation_results.json                Fusion ablation results (4 configurations)
├── deberta_standalone_eval.json         DeBERTa standalone evaluation results
│
├── sightengine_cache.json               SightEngine API response cache
├── hive_cache.json                      Hive API response cache
├── blip_itm_results.png                 BLIP ITM training curves
├── blip2_results.png                    BLIP-2 evaluation results
├── fusion_classifier_results.png        Fusion v1 results chart
├── fusion_3module_finetuned_results.png Fusion v3 ablation chart
├── results_summary.html                 Full HTML results summary (all tables)
├── thesis_writing_notes_v2.md           Complete thesis writing notes + paragraphs
│
├── kaggle_dataset_minimal.zip           Pre-built Kaggle upload (5.5k images)
├── kaggle_dataset_full.zip              Pre-built Kaggle upload (71k images)
│
└── dataset/
    ├── data/NewsClipPings/
    │   ├── merged_balanced/             Labels (train/val/test JSON)
    │   └── metadata/                    Captions + image paths
    ├── origin/origin/                   Raw images (bbc, guardian, usa_today, washington_post)
    └── MMFakeBench/                     MMFakeBench dataset
```
