# Pics Can Lie — Full Project Summary
> Multimodal Misinformation Detection — Thesis Project
> Last updated: June 4, 2026 (Honest audit COMPLETE + isotonic calibration shipped — NewsCLIPpings test: **AUC 0.932 / acc 0.865** at a blind, val-derived calibrated operating point)

> ⚠️ **READ FIRST — this summary was rewritten after an honesty audit.** Several earlier
> headline numbers were withdrawn: the **86.48%** NewsCLIPpings test figure was produced by
> sweeping the decision threshold on the **test labels** (data snooping) on top of a broken
> feature path (the train scaler was never applied). The honest, audited pipeline reports
> **AUC as the primary metric** and accuracy only at a threshold/calibrator fit on non-test
> data. The single source of truth for numbers is `docs/corrected_results.md` and
> `docs/comparison_table.md`; this file is the narrative companion. Old inflated claims are
> struck through or labelled "(withdrawn)".

---

## 1. Project Overview

**Goal:** Detect out-of-context news image-caption pairs using a multi-signal fusion pipeline.

**Task:** Binary classification — REAL (image matches caption) vs FAKE (image is out-of-context).

**Primary dataset:** NewsCLIPpings Merged/Balanced — ~71k train, 5k val, 7264 test, binary labels.

**Secondary dataset:** MMFakeBench — 1000 val samples across 4 fake categories.

### Honest headline result (NewsCLIPpings test, 7264 samples)
| Metric | Value | Operating point |
|---|---|---|
| **AUC (primary)** | **0.932 [0.926, 0.937]** | threshold-free |
| Accuracy (production) | **0.865 [0.856, 0.873]** | **isotonic-calibrated @0.5** (val-derived, blind) |
| Accuracy (superseded) | 0.831 [0.822, 0.839] | raw frozen 0.5523 (blind) |
| ~~Accuracy (invalid)~~ | ~~0.8648~~ | threshold swept on TEST labels — **withdrawn** |

- **The lift 0.831 → 0.865 is honest calibration, not snooping.** The raw AITR fused prob is
  over-confident (mean confidence 0.96 vs accuracy 0.83, **ECE 0.138**); a **val-fit isotonic
  calibrator** (ECE → 0.017) lets the verdict be taken at a clean, transferable 0.5. AUC is
  unchanged (calibration is monotonic). This recovered ~96% of the gap to the test-label
  thresholding ceiling (0.866) — i.e. the AUC/accuracy gap was a **threshold/calibration
  problem, not model capacity**. See Section 17 and `results/calibration_threshold_report.json`.
- Per-source test acc @ global isotonic: BBC 0.845 · Guardian 0.863 · USA Today 0.877 · Wash Post 0.870.
- **Source-aware thresholds are a FAILED ABLATION (not shipped):** val-derived, shrunk toward
  global (k=1000), they scored 0.8652 vs global isotonic 0.8647 — within noise. We ship the
  simpler global isotonic calibrator. (See Section 6.)
- Methods note: test has no retrieved evidence, so the 7 evidence/NLI scalars are imputed to
  TRAIN means for **both** the val calibration set and test (matched pipelines); this is why the
  matched-pipeline val AUC is **0.924** (vs 0.948 when val uses its real evidence scalars).

**MMFakeBench generalization (honest framing — see Section 10/16):**
- Best **deployable** model: GBM on [clip_prob, clip_sim, ateeq] — **AUC 0.792**, only ~9pp over
  the always-Fake floor (0.700). Shortcut-free (no Ateeq) it drops to **AUC 0.706**, near the floor.
- The **Wikipedia-NLI veto "79.80%" is an ORACLE** (it gates on the ground-truth `original`
  category, unavailable at inference) — **not deployable**. The deployable category-agnostic
  version has precision 0.23 / acc 0.537. (The earlier "beats MIRAGE" claim is withdrawn.)
- The Ateeq *fusion* lift on MMFakeBench is not robust (drops to the floor without it). BUT a
  **wild-image test corrected the "shortcut" label**: as a standalone AI-image detector Ateeq
  **generalizes** to unseen generators (StyleGAN 1.00 / Midjourney 0.85 recall ≥ in-distribution
  Fakeddit 0.67; wild AUC 0.864 ≈ control 0.869) — **not** a pool shortcut — though it has a
  resolution bias (false positives on large real photos). See Section 4.7 / 10.3. It is the
  **demo's image-origin signal** (replaced SightEngine).

**Literature targets — DO NOT claim "beaten"; verify protocol, lead with AUC (see comparison_table.md):**
- SNIFFER: 88.40% (external entity/LLM retrieval) — our 0.865 is ~1.9pp below, but **internal-only
  vs external-retrieval**, not like-for-like. Reframe as "competitive, no external API."
- MIRAGE: 75.10% (GPT-4o-mini + web retrieval) — our deployable MMFakeBench number is near the floor;
  the 79.80% that "beat" it was an oracle (withdrawn).
- MUSE-MLP 90.00% / MUSE-AITR 93.30% / RED-DOT 90.30% — all use external evidence; confirm split/protocol.

---

## 2. Hardware Setup — Two Machines

This project runs across TWO machines. Critical context for understanding which files are where.

### Machine A — Personal laptop (RTX 4060 Laptop)
- **Drive:** `D:\Pics Can Lie\` and `E:\Pics Can Lie\`
- **GPU:** NVIDIA RTX 4060 Laptop (8GB VRAM)
- **Used for:** Evidence pipeline (has good internet), DeBERTa/Wiki NLI scoring, fusion experiments, threshold tuning, MMFakeBench evaluation
- **Key files here:**
  - `D:\Pics Can Lie\links_val.json` — evidence links from Abdelnabi et al.
  - `D:\Pics Can Lie\deberta_val_scores_v2.csv` — DeBERTa NLI scores (5000 samples)
  - `D:\Pics Can Lie\evidence_clip_scores.csv` — evidence CLIP scores (s2-s6, 5000 samples, 95.8% coverage)
  - `D:\Pics Can Lie\wiki_nli_scores.csv` — Wikipedia NLI scores (5000 samples)
  - `D:\Pics Can Lie\wiki_page_cache.json` — cached Wikipedia pages (3245 entries)
  - `D:\Pics Can Lie\clip_finetuned_v2\val_features\` — copied from K:\, all 5 files
  - `D:\Pics Can Lie\fusion_aitr\aitr_weights.pt` — trained AITR model weights
  - `D:\Pics Can Lie\fusion_aitr\optimal_threshold.json` — optimal threshold = 0.54 global
  - `D:\Pics Can Lie\tta_img_features.pt` — TTA features (computed, not useful)
  - `E:\Pics Can Lie\dataset\MMFakeBench\` — MMFakeBench val set (1000 samples, extracted)

### Machine B — 4090 workstation
- **Drive:** `K:\Joee El Ghandour\Pics Can Lie\`
- **GPU:** NVIDIA RTX 4090 (22.5GB VRAM)
- **Used for:** CLIP fine-tuning (v1, v2, v3, v4 attempts)
- **Key files here:**
  - `clip_finetuned_v2\` — CLIP v2 best (85.6%), 2350MB checkpoint
  - `clip_finetuned_v3\` — CLIP v3 CrossModalAttention attempt (85.3%, worse)
  - `clip_finetuned_v4\` — CLIP v4 attempt (84.07% at epoch 6, crashed)
  - `clip_finetune_v4_fixed.ipynb` — fixed v4 notebook with AMP + resume logic
  - `resume_training.py` — standalone script for background training
  - `training_summary_v4.txt` — per-epoch summary (monitor with Get-Content -Wait)

**IMPORTANT:** When a notebook says `K:\` it runs on the 4090. When it says `D:\` it runs on personal machine. Do NOT mix paths. The 4090 has BAD internet — run evidence pipeline on personal machine instead.

---

## 3. Folder Structure (Personal Machine D:\)

```
D:\Pics Can Lie\
├── kaggle_dataset_full\
│   ├── images\                    ← news images
│   ├── merged_balanced\           ← train.json, val.json, test.json
│   └── metadata\                  ← train.json, val.json (has caption_entities_rel, etc.)
│       NOTE: article .txt files DO NOT EXIST — only JSON + images downloaded
├── models\clip\                   ← CLIP ViT-L/14 weights (~890MB)
├── clip_finetuned_v2\
│   ├── clip_classifier.pt         ← 2.35GB, NOW COPIED TO D:\ (epoch 14, F1=0.8522, acc=0.8556)
│   └── val_features\              ← copied from K:\
│       ├── clip_finetuned_probs.npy   ← (5000,) ← USE FOR FUSION
│       ├── clip_finetuned_sims.npy    ← (5000,) ← USE FOR FUSION
│       ├── clip_img_features.pt       ← (5000, 768) NOT L2-normalized as saved
│       ├── clip_txt_features.pt       ← (5000, 768) NOT L2-normalized as saved
│       └── val_sample_ids.csv         ← (5000 rows) MASTER ID FILE
├── ai_detector_finetuned\         ← FINE-TUNED ATEEQ (for MMFakeBench)
│   ├── model.safetensors          ← 354MB SiglipForImageClassification
│   ├── config.json
│   └── preprocessor_config.json
├── finetune_ai_detector.py        ← Ateeq fine-tuning script
├── run_ateeq_fusion.py            ← Ateeq fusion runner
├── mmfakebench_ai_scores_finetuned.csv ← Ateeq scores (200 samples: original + visual_vd)
├── ablation_ateeq_vs_sightengine.json  ← AI module ablation (300 samples)
├── fusion_features_300_ateeq.csv  ← fusion features with ateeq_score column
├── deberta_val_scores_v2.csv      ← (5000 rows, 100% aligned to v2 IDs)
├── evidence_clip_scores.csv       ← (5000 rows, s2-s6 means)
├── wiki_nli_scores.csv            ← (5000 rows, wiki_score/min/max/entity_count/coverage)
├── wiki_page_cache.json           ← (3245 Wikipedia pages cached)
├── geo_scores.csv                 ← StreetCLIP geolocation (FAILED — diff <0.004, can delete)
├── tta_img_features.pt            ← TTA features (FAILED -0.30%, can delete)
├── fusion_aitr\
│   ├── aitr_weights.pt            ← trained AITR transformer weights
│   ├── optimal_threshold.json     ← {threshold: 0.54, accuracy: 0.8830}
│   └── results.json               ← AITR results summary
└── mmfakebench_results.json       ← MMFakeBench eval results (now complete)
└── test_set_results\
    └── test_results.json          ← OLD (86.48%, data-snooped — withdrawn). Use results/test_set_results/test_results_fixed.json

E:\Pics Can Lie\dataset\MMFakeBench\
├── MMFakeBench_val\               ← 1000 val images EXTRACTED (300 real, 700 fake)
├── MMFakeBench_val.json           ← val annotations (gt_answers uses 'Fake' not 'False'!)
├── MMFakeBench_test.json          ← test annotations (3065KB — larger split for TRAINING)
├── MMFakeBench_test.zip           ← test images (6.6M — NOT yet extracted, needed for training)
└── MMFakeBench_val.zip            ← already extracted
```

---

## 4. All Models Tried — Results

### 4.1 BLIP ITM Base v1-v2
- **Result:** Val Acc = 75-76%
- **Status:** Superseded

### 4.2 BLIP ITM Large v4 (4090)
- **Config:** Vision layers 18-23 + text 6-11 unfrozen, LR 3e-6, batch 32
- **Result:** Val Acc = 78.00%, F1 = 0.7794
- **Critical:** CLS embeddings NOT discriminative (F1=0.46). Only ITM scalar is useful.

### 4.3 Fine-tuned CLIP ViT-L/14 v1 (4090)
- **Result:** Val Acc = 83.58%, F1 = 0.8356

### 4.4 Fine-tuned CLIP ViT-L/14 v2 (4090) ← CURRENT BEST CLIP
- **Config:** Head LR=5e-5, encoder LR=5e-7 (÷100), batch 64, dropout=0.5, BatchNorm
- **Key insight:** 100× LR ratio between head and encoder was decisive
- **Result:** Val Acc = 85.60%, F1 = 0.852

### 4.5 Fine-tuned CLIP ViT-L/14 v3 (4090) — WORSE THAN V2
- **Config:** Same as v2 + CrossModalAttention + LabelSmoothBCE
- **Result:** Val Acc = 85.30%, F1 = 0.8525 — marginally worse than v2
- **Why worse:** CrossModalAttention added complexity without benefit; slow convergence

### 4.6 Fine-tuned CLIP ViT-L/14 v4 (4090) — INCOMPLETE
- **Config:** Full 71k dataset, hard negative mining, contrastive loss (disabled), layer-wise LR decay, AMP
- **Status:** Crashed at epoch 6 (best F1=0.8372, 84.07%) — worse than v2 at this point
- **Training issues:**
  - `num_workers=4` with `pin_memory=True` → deadlock on Windows Jupyter
  - Contrastive loss (InfoNCE temp=0.07) produced loss=4.0 dominating training — disabled
  - Background script (`resume_training.py`) also crashed
  - `if __name__ == '__main__':` guard required for num_workers>0 on Windows
- **Current state:** 6 epochs complete, still improving but didn't finish
- **To resume:** Run `train_v4_final.py` with `num_workers=0, pin_memory=False`

### 4.7 Fine-tuned Ateeq AI Detector (4060, for MMFakeBench only)
- **Base model:** `Ateeqq/ai-vs-human-image-detector` (SiglipForImageClassification)
- **Script:** `finetune_ai_detector.py` (on D:\ and E:\)
- **Output model:** `D:\Pics Can Lie\ai_detector_finetuned\model.safetensors` (354MB)
- **Two-phase training:** Phase 1 — classifier head only, 3 epochs, LR 1e-4. Phase 2 — classifier + last 2 encoder layers, 3 epochs, LR 1e-5.
- **Training data (800 total, 80/20 split):** 500 real NewsCLIPpings images (bbc/guardian/usa_today/wash) + 250 AI-generated (antifact) + 250 PS-edited (Fakeddit)
- **Purpose:** Reduce false positives on real news photos while detecting AI/manipulated images. For MMFakeBench ONLY — NOT used on NewsCLIPpings (real photos give zero variance).
- **Before fine-tuning:** 56% overall on MMFakeBench, 48% on visual_veracity_distortion (worse than random)
- **After fine-tuning:** 75.5% overall, **94% on visual_veracity_distortion at thr=0.20** — huge improvement
- **On NewsCLIPpings (300-sample ablation):** real mean=0.407, fake mean=0.438, diff=+0.030 (noise — confirms it hurts here)
- **Output scores file:** `mmfakebench_ai_scores_finetuned.csv` — columns include `ai_score_ft`, `predicted_label_ft`. Covers 200 samples (100 original + 100 visual_vd).

**Wild-image sanity test (June 2026) — is Ateeq a genuine detector or a source-pool shortcut?**
Scripts: `scripts/collect_wild_ai_test.py`, `scripts/ateeq_wild_test.py`. Outputs:
`results/ateeq_wild_test.csv`, `results/ateeq_wild_test_summary.json`.
- Sets: WILD AI = StyleGAN2 (tpdne) + Midjourney v6 (HF) [out-of-distribution] + MMFB visual_vd
  Fakeddit edits [in-distribution]; WILD REAL = Pascal-VOC + Imagenette + Wikimedia; CONTROL =
  MMFB original-reals + visual_vd AI (disjoint from the injected ones).
- **Result — GENUINE DETECTOR, generalizes, with a resolution confound:**
  - Per-source recall@0.5: **StyleGAN 1.00, Midjourney 0.85** ≥ in-distribution **Fakeddit 0.67**.
    Unseen generators detected ≥ training pool → **not** a source-pool shortcut.
  - Wild AUC (external) = **0.864** ≈ control AUC **0.869**; control reproduced ~94% (48/50 recall@0.2).
  - Confound: `ai_score` r ≈ **0.6** with resolution/file-size on real images; all confident errors
    were large high-res real photos; within-real source-separation AUC peaks 0.75 (Wikimedia).
  - Caveat: small n (17 real; 2 external generator families, StyleGAN faces-only). Directional, not bulletproof.
- **Implication:** the earlier "non-generalizable shortcut" claim is corrected to "genuine AI
  detector that generalizes, but resolution-biased on real images." Still NOT useful on
  NewsCLIPpings (authentic photos only) — its role is **image-origin for wild demo uploads**.

### Ablation: AI module choice on NewsCLIPpings (300 samples, 5-fold CV)
File: `ablation_ateeq_vs_sightengine.json`
| Combination | Accuracy | F1 |
|---|---|---|
| BLIP only | 82.67% | 0.8256 |
| DeBERTa only | 58.00% | 0.6795 |
| BLIP + SightEngine | 88.00% | 0.8809 |
| BLIP + Ateeq | 82.33% | 0.8227 (Ateeq HURTS) |
| BLIP + DeBERTa | 85.00% | 0.8484 |
| **BLIP + DeBERTa + SightEngine** | **89.67%** | **0.8958** |
| BLIP + DeBERTa + Ateeq | 85.33% | 0.8512 |
Confirms: Ateeq adds noise on NewsCLIPpings real photos; SightEngine was the better AI signal there but isn't used in the final NewsCLIPpings system (CLIP-only visual signal won).

---

## 5. Signal Investigation Results

### 5.1 Evidence Pipeline (s2-s6)
- **Coverage:** 4790/5000 samples (95.8%) got at least one evidence image
- **Scores (non-zero):**
  - s2 (orig vs direct): mean=0.731, count=4763
  - s3 (caption vs direct): mean=0.235, count=4763
  - s4 (orig vs inverse): mean=0.620, count=3205
  - s5 (caption vs inverse): mean=0.188, count=3205
  - s6 (cross-evidence): mean=0.573, count=3178
- **Key finding:** s2-s3 mismatch (the theoretically correct OOC signal) shows diff=0.001 between real/fake — NOT discriminative. Evidence links are article-text-based, not image-specific.
- **Contribution:** ~+1.8% over CLIP alone when combined in fusion

### 5.2 DeBERTa NLI
- **File:** `deberta_val_scores_v2.csv`
- **Distribution:** Bimodal — near 0 or near 1, almost nothing in between
- **Real vs fake diff:** +0.0066 — essentially zero
- **Root cause:** Same article ID appears as both label 0 and label 1 (NewsCLIPpings creates fakes by swapping captions between real articles). DeBERTa scores article text, not image content. Both real and fake versions of an article have similar entailment with the caption.
- **XGBoost importance:** 0.0247 — nearly useless
- **Conclusion:** Text-only NLI cannot discriminate OOC on NewsCLIPpings

### 5.3 Wikipedia NLI
- **File:** `wiki_nli_scores.csv`
- **Method:** Extract entities from `caption_entities_rel` (already Wikipedia-normalized), fetch summaries, run DeBERTa NLI
- **Coverage:** 4614/5000 samples with wiki pages
- **Real vs fake diff:** -0.002 — essentially zero
- **Root cause:** Same as DeBERTa — captions are factually correct. The fake is in the image pairing, not the caption text. Wikipedia can't detect that.
- **Conclusion:** External knowledge signals are fundamentally ineffective on this dataset

### 5.4 Key Finding (thesis contribution)
**Text-based signals (DeBERTa, Wikipedia NLI, feature engineering) are all ineffective on NewsCLIPpings because captions are sourced from factually accurate articles. The manipulation is purely in the image-caption pairing — not detectable from caption text alone. Visual signals are exclusively discriminative.**

---

## 6. Fusion Architecture Results

| Fusion | Accuracy | Notes |
|---|---|---|
| MLP (3 signals: CLIP+DeBERTa) | 87.40% | Baseline fusion |
| XGBoost (8 signals) | 87.30% | Same as MLP |
| XGBoost + wiki (13 signals) | 87.20% | Wiki adds noise |
| Feature engineering (37 feats) | 86.20% | WORSE — overfitting |
| AITR transformer (768-dim tokens) | AUC 0.948 (val) | Best architecture; small edge over scalar-only fusion (0.930) |

> ⚠️ The accuracy column above is from the old val-only, threshold-tuned protocol and is
> superseded. Report **AUC** (val 0.948 / test 0.932) and the calibrated test accuracy (0.865).

### AITR Architecture
- 5 tokens: img_emb(768), txt_emb(768), img*txt(768), img-txt(768), scalar_proj(768)
- Learnable CLS token attends across all 5 via self-attention (8 heads, 2 layers)
- Scalar signals: [clip_prob, clip_sim, deberta, s2, s3, s4, s5, s6, wiki_score]
- Trained on val split (80/20), best F1=0.8799 at optimal threshold

### Source-Aware Thresholds — FAILED ABLATION (withdrawn from the final system)
The earlier per-source thresholds (BBC 0.58 / WashPost 0.57 / Guardian 0.54 / USAToday 0.44)
were tuned and evaluated on the same val set — overfitting. The **honest** redo derives them on
val only and **shrinks each toward the global threshold** by sample count (pseudo-count k=1000)
to tame small-source overfit (e.g. the raw Washington Post val optimum was 0.127 on n=817 vs a
global ~0.02). Applied **blind** to test they scored **0.8652 [0.857, 0.873]** — statistically
indistinguishable from the global isotonic operating point (**0.8647**). **Decision: ship the
global isotonic calibrator; report source-aware only as an ablation that did not beat it.**
Persisted (for the record, not shipped): `models/fusion_aitr/source_aware_thresholds.json`.

### Shipped operating point (final system)
Global **isotonic calibration** of the AITR fused prob, decided at **0.5**
(`models/fusion_aitr/calibrators.joblib`, fit on the 5000-row val split). Raw AITR is
over-confident (ECE 0.138 → 0.017 after isotonic). Test acc 0.865, AUC 0.932 unchanged.

---

## 7. Full Accuracy Progression (thesis table)

> Numbers below are accuracy-only and from mixed protocols; treat as historical context.
> The defensible comparison leads with **AUC** at a frozen/calibrated (blind) threshold.

| System | Metric | Evidence source |
|---|---|---|
| NewsCLIPpings baseline | 72.44% acc | No evidence |
| VERITE (CLIP ViT-L/14) | 74.40% acc | No evidence |
| Our BLIP ITM large | 78.00% acc | No evidence |
| COSMOS | 85.00% acc | No evidence |
| Our fine-tuned CLIP v2 (standalone) | 85.60% acc | No evidence |
| SNIFFER | 88.40% acc | External entity/LLM retrieval |
| **Ours (CLIP→AITR, honest)** | **AUC 0.932 / acc 0.865** (calibrated, blind) | Internal only, no external API at test |
| MUSE-MLP | 90.00% acc | Google API evidence |
| RED-DOT | 90.30% acc | Evidence re-ranking |
| MUSE-AITR | 93.30% acc | Google API evidence |

**Honest framing:** ours is **competitive with no external API** — within ~1.9pp of SNIFFER's
reported accuracy while SNIFFER uses external retrieval, and we lead with **AUC 0.932** where
protocols differ. The earlier "**you beat SNIFFER**" and "$0 vs $2,000" headlines are
**withdrawn** as not like-for-like (and the 88.50%/86.48% numbers they relied on were tuned/snooped).

---

## 8. Post-Processing Attempts (all tried, none helped)

| Technique | Result | Why it failed |
|---|---|---|
| TTA (5 augmentations) | -0.30% | News photos need full context; crops remove discriminative regions |
| XGBoost + AITR ensemble | -0.40% | Mixing weaker model with stronger always pulls result down |
| Platt calibration | -0.30% | Only 700 cal samples — not enough for meaningful calibration |
| Feature engineering (RED-DOT-style) | -1.10% | RED-DOT applied ops on 768-dim vectors, not scalar aggregations |
| Global threshold tuning (0.54) | +0.80% | Only post-processing that helped |
| Source-aware thresholds | +0.20% additional | BBC 0.58 was the key change |

---

## 9. Error Analysis Findings

- **Total errors:** 117/1000 (11.7%) at threshold 0.54
- **FP (real→fake):** 71 — model is trigger-happy, calls real images fake
- **FN (fake→real):** 46 — misses some fakes

**By source:**
| Source | Samples | Error rate |
|---|---|---|
| BBC | 139 | **17.99%** — hardest |
| Washington Post | 157 | 15.29% |
| Guardian | 474 | 10.13% |
| USA Today | 230 | **8.70%** — easiest |

**Why BBC is hardest:** BBC covers politically-charged international news. Images of politicians in different geographic locations are nearly identical visually — CLIP can't distinguish "politician in London" from "politician in Aberdeen."

**High-confidence error pattern:** Most confident wrong predictions are politics-related (Ed Balls, Aberdeen, shadow chancellor). These represent a fundamental failure mode — geographically-grounded claims cannot be verified from image content alone.

---

## 10. MMFakeBench Evaluation

**Dataset:**
- Val: 1000 samples across 4 categories. **CONFIRMED LABEL STRUCTURE (verified May 25, 2026):**
  - `original`: 300 samples — **ALL REAL** (gt_answers='True'). Zero fake samples.
  - `mismatch`: 300 samples — **ALL FAKE** (gt_answers='Fake').
  - `textual_veracity_distortion`: 300 samples — **ALL FAKE**.
  - `visual_veracity_distortion`: 100 samples — **ALL FAKE**.
  - Always-Fake floor = 700/1000 = **70.00%** exactly.
- Train (from `MMFakeBench_test.json`): 10,000 samples, 70/30 fake/real. Extracted and verified intact.

**CRITICAL LABEL NOTE:** MMFakeBench uses `gt_answers == 'Fake'` (NOT 'False'). Conversion: `(df['gt_answers'] == 'Fake').astype(int)`. Using 'False' → all-zero labels. This bug has happened twice — verify on every new script.

---

### 10.1 Zero-shot baseline (NewsCLIPpings-trained AITR on MMFakeBench val)
- Overall: 58.5% at Ateeq thr=0.50 (AITR thr=0.05) — best zero-shot operating point
- AUC-ROC = 0.6478 (real discriminative ability, miscalibrated for 70/30)
- Per-category:
  - `original`: 65% (Ateeq false-positives on real news photos)
  - `mismatch`: 59% (OOC task transfers but domain-shifted)
  - `textual_veracity_distortion`: 40% (worse than random — CLIP can't see text manipulation)
  - `visual_veracity_distortion`: 93% (Ateeq excels)

This number is BELOW the always-predict-Fake floor (70%), which is why retraining was necessary.

---

### 10.2 Step 1 — Retrained AITR on MMFakeBench (COMPLETE)

**Setup:**
- Warm-started from `fusion_aitr/aitr_weights.pt` (NewsCLIPpings AITR). Reinit ONLY scalar_proj layer (9 → 3 scalars).
- Train: 80/20 stratified split of MMFakeBench_test.json's 10K samples (random_state=42).
- Scalars: `[clip_prob, clip_sim, ateeq_score_ft]` — CLIP v2 outputs + fine-tuned Ateeq.
- Loss: BCEWithLogitsLoss with `pos_weight = n_real/n_fake = 0.4286` (counteracts 70/30 skew).
- AdamW: lr=1e-4 on `scalar_proj`, lr=1e-5 on warm-started weights. Cosine LR + 1-epoch warmup. Batch 64.
- Early stopping: converged at **epoch 1** (best internal-val F1=0.795, AUC ~0.82), stopped after epoch 5.

**Results — held-out val (1000 samples):**

| | Zero-shot | Retrained (Step 1) | Δ |
|---|---:|---:|---:|
| **Overall accuracy** | 58.5% | **72.6%** | **+14.1** |
| AUC-ROC | 0.6478 | 0.7367 | +0.089 |
| `original` (300, real) | 65% | 59% | −6.0 |
| `mismatch` (300, fake) | 59% | 72% | +13.0 |
| `textual_veracity_distortion` (300, fake) | 40% | **79.7%** | **+39.7** |
| `visual_veracity_distortion` (100, fake) | 93% | 94% | +1.0 |

**Best threshold = 0.30.** At thr=0.50 the model is at 69.5% (below always-Fake). The +3.1% from 0.50→0.30 is threshold calibration; the +11% over zero-shot is genuine learning of MMFakeBench's distribution. Clears the always-Fake floor (70%) by +2.6% → confirms real signal, not just class-prior exploitation.

**OR-fusion (Ateeq | AITR) is WORSE than AITR-only:** 71.4% vs 72.6%. Ateeq's signal is already inside AITR via the scalar token; OR'ing it on top double-counts and flips real-image predictions to fake. **Use AITR-only @ thr=0.30, do not OR with raw Ateeq.**

### 10.3 Sanity checks on the Step 1 result

Two diagnostics run post-training (preserved in `mmfakebench_training/sanity_textual_vd.py`):

**Check 1 — textual_vd signal is real:**
- clip_prob Δ(fake − real) = +0.127 — above the 0.10 bar
- AITR median output: fake=0.66, real=0.24 — clean separation
- → The textual_vd gain (+39.7) is genuine, not a threshold artifact

**Check 2 — Ateeq is doing some of the work via a shortcut:**
- Ateeq scores `textual_vd` real images at 0.66 ai-ness vs `original` real images at 0.41 — a +0.26 gap
- This shouldn't exist for an image-only detector on real photos
- Ateeq has latched onto **image-source distribution** (probably resolution/compression/aspect-ratio signatures across MMFakeBench's source pools) rather than fakeness
- Part of the +39.7 headline leans on this shortcut. Not a measurement error, but a generalization caveat that must be documented in the thesis

**Check 3 — 61 textual_vd false negatives are unimodal mid-band (0.10–0.20):**
- Model is uncertain on these, not blind
- Threshold sweep already found the optimum; recovering them requires NEW features, not more tuning
- → Justifies Step 2 (DeBERTa NLI)

### 10.4 Step 2 — DeBERTa NLI Ablation (COMPLETE)

**Goal:** Test whether adding DeBERTa NLI to AITR fusion lifts MMFakeBench accuracy.

**4-variant ablation (trained on 80/20 stratified split of 10K, random_state=42):**

| Variant | Scalars | Overall | AUC | original | mismatch | textual_vd | visual_vd |
|---|---|---|---|---|---|---|---|
| A | clip_prob, clip_sim | 70.00% ⚠ | 0.695 | 0% (degenerate) | 100% | 100% | 100% |
| B | + ateeq | **72.60%** | 0.737 | 59.00% | 72.00% | 79.67% | 94.00% |
| C | + deberta | 71.70% | 0.694 | 15.67% | 95.67% | 94.67% | 99.00% |
| D | + ateeq + deberta | 72.40% | 0.742 | 59.33% | 71.67% | 79.33% | 93.00% |

**Analysis numbers:**
1. Ateeq shortcut (B.tvd − A.tvd): uninterpretable — A degenerated to always-Fake
2. NLI legitimate contribution (C.tvd − A.tvd): uninterpretable — same reason
3. Additivity (D − max(B,C)): −0.20pp → NLI is redundant with Ateeq for textual_vd

**Key findings:**
- Variant A collapses to always-Fake at all tested thresholds — CLIP alone insufficient without Ateeq on MMFakeBench
- Variant C (deberta, no ateeq): high textual_vd/mismatch but destroys original (15.67%) — DeBERTa biases toward FAKE
- Variant B (ateeq only) = best balance: 72.60%, solid per-category
- Variant D (ateeq + deberta): near-identical to B overall, DeBERTa adds no lift when Ateeq present
- **Documented negative finding:** Entity-grounded Wikipedia NLI does not improve over Ateeq alone on MMFakeBench. Writing-style mismatch (encyclopedia vs news) + 42% zero-coverage rate explain the failure.
- **Best MMFakeBench AITR checkpoint: `aitr_mmfb_best.pt` = Variant B** (72.60%)

### 10.5 Files state (COMPLETE as of May 25, 2026)

```
D:\Pics Can Lie\mmfakebench_training\
├── extract_test_split.py                 ✅ 1.1
├── compute_clip_features_mmfb.py         ✅ 1.2
├── compute_ateeq_train.py                ✅ 1.3
├── train_aitr_mmfb.py                    ✅ 1.4
├── evaluate_aitr_mmfb.py                 ✅ 1.5
├── smoke_clip_features.py                ✅ smoke test
├── sanity_textual_vd.py                  ✅ Step 1 post-hoc diagnostics
├── extract_caption_entities_mmfb.py      ✅ 2.1
├── fetch_wiki_pages_mmfb.py              ✅ 2.2 (bulk API)
├── compute_deberta_nli_mmfb.py           ✅ 2.3
├── train_aitr_mmfb_v2.py                 ✅ 2.4 — COMPLETE
├── evaluate_ablation_mmfb.py             ✅ 2.5 — COMPLETE
├── aitr_mmfb_v2_A.pt                     ✅ Variant A (degenerate baseline)
├── aitr_mmfb_v2_B.pt                     ✅ Variant B (best: 72.60%)
├── aitr_mmfb_v2_C.pt                     ✅ Variant C (deberta, no ateeq)
├── aitr_mmfb_v2_D.pt                     ✅ Variant D (ateeq + deberta)
├── ablation_results_v2.json              ✅ full ablation table
├── train_features/, val_features/        ✅ CLIP feats
├── train_ateeq_scores.csv (10K rows)     ✅
├── val_ateeq_scores_full.csv (1K rows)   ✅
├── deberta_nli_train.csv (10K rows)      ✅
├── deberta_nli_val.csv (1K rows)         ✅
├── caption_entities_{train,val}.json     ✅
├── aitr_mmfb_best.pt                     ✅ Step 1 = Variant B (72.60%)
├── training_log.csv                      ✅
├── eval_results_aitr_only.json           ✅ 72.6%
└── RESULTS.md                            ✅ Step 1 writeup

D:\Pics Can Lie\overnight_v2\             ✅ NEW — overnight analysis results
├── task1_baseline.json                   ✅ full threshold sweep
├── task2_per_cat_thresholds.json         ✅ ⚠ OVERFIT — do not use (see Section 16)
├── task3_attention.json                  ✅ attention weights (scalars dominate)
├── per_sample_attention_mmfb.csv         ✅ per-sample attention breakdown
├── task4_ensemble.json                   ✅ B+C ensemble results
├── task5_wiki_veto.json                  ✅ Wikipedia veto results
└── DONE_v2.md                            ✅ overnight run summary

D:\Pics Can Lie\wiki_page_cache.json      ✅ ~16K entries
```

---

## 11. What Was NOT Done and Why

| Thing | Why not done |
|---|---|
| BLIP-2 / ViT-G | Needs 35-40GB VRAM — impossible on 4090 |
| Full CLIP unfreezing | 119.5M params, overfitting on 71k samples |
| Google API evidence | ~$2,000 to reproduce. Not feasible. |
| Ateeq on NewsCLIPpings | Real photos — detector outputs constant "not AI" = zero variance |
| Article text for NLI | .txt files don't exist locally, only JSON+images downloaded |
| Cross-attention fusion (v3) | Added complexity without benefit (85.3% vs 85.6%) |
| MAD-Sherlock LLM debate | API cost + complexity; would give +2-5% but expensive |
| LAMAR synthetic data | Complex to implement; 3-5 days of work |
| MMFakeBench training | NOT YET done — current 58.5% is zero-shot. PLANNED next step (see Section 10) |
| Claude API explanations | Planned, not implemented — feeds logged scalar scores to Claude for rationale |

---

## 12. Key Technical Notes

### CLIP loading (mandatory)
```python
clip_model, clip_preprocess = clip.load('ViT-L/14', device=device, jit=False)
clip_model = clip_model.float()   # REQUIRED — prevents LayerNorm RuntimeError
# jit=False — prevents MemoryError on Windows
```

### CLIP embeddings alignment
```python
# clip_img_features.pt and clip_txt_features.pt are NOT L2-normalized as saved
img_feats = F.normalize(torch.load('clip_img_features.pt'), dim=-1)
txt_feats = F.normalize(torch.load('clip_txt_features.pt'), dim=-1)
```

### Evidence ID mapping (critical)
```python
# links_val.json uses integer index keys, NOT article IDs
idx_to_id   = {str(i): str(a['id']) for i, a in enumerate(annotations)}
id_to_links = {aid: links_data[idx] for idx, aid in idx_to_id.items() if idx in links_data}
```

### Windows DataLoader (always use)
```python
# num_workers > 0 deadlocks in Jupyter on Windows — use 0
DataLoader(dataset, batch_size=64, num_workers=0, pin_memory=False)
# For standalone .py scripts with num_workers > 0:
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    # ... all training code here
```

### DeBERTa output order — VERIFY PER MODEL (this changed)
```python
# cross-encoder/nli-deberta-v3-large — empirically verified May 2026:
# output[0]=contradiction, output[1]=ENTAILMENT, output[2]=NEUTRAL
# This is NOT the conventional [contradiction, neutral, entailment] order.
# ALWAYS run a sanity pair before scoring:
#   premise=hypothesis="The sky is blue." → entailment column should be ~0.99
# If your existing deberta_val_scores_v2.csv was computed with index 2,
# it actually contains NEUTRAL scores, not entailment — re-verify.
entailment_score = softmax(logits)[1]   # NOT [2]
```

### CLIP v4 training on Windows (AMP required)
---

## 16. Post-Fusion Analysis — Overnight Run Results (May 25, 2026)

### 16.1 Wikipedia NLI Veto — NEW HEADLINE RESULT ✅

**Finding:** AITR Variant B over-predicts FAKE on `original` samples (59.00% accuracy). Wikipedia NLI entailment score > 0.05 on an `original` sample predicted FAKE is a perfect-precision veto signal.

**Why it works:**
- `original` category = 300 samples, ALL REAL (gt_answers='True') — verified
- AITR misclassifies 119/300 as FAKE at thr=0.30
- Of those 119, 76 have deberta_score > 0.05 → override to REAL
- Precision of veto: **76/76 = 100%** (zero wrong overrides, impossible since original=all REAL)
- Remaining 45 hard negatives: mean deberta=0.0102, mean AITR prob=0.774 — not recoverable

**Optimal veto config:**
```python
# Apply AFTER AITR inference, on original category only
if fake_cls == 'original' and aitr_pred == FAKE and deberta_score > 0.05:
    final_pred = REAL  # override
```

**Results:**

| System | Overall | original | mismatch | textual_vd | visual_vd |
|---|---|---|---|---|---|
| Always-Fake floor | 70.00% | 0% | 100% | 100% | 100% |
| AITR Variant B | 72.60% | 59.00% | 72.00% | 79.67% | 94.00% |
| **+ Wikipedia veto (deb>0.05)** | **79.80%** | **86.00%** | 72.00% | 79.67% | 94.00% |
| MIRAGE (GPT-4o-mini) | 75.10% | — | — | — | — |

~~**79.80% beats MIRAGE (75.10%) by +4.70pp at zero inference cost.**~~ **WITHDRAWN:** the
79.80% relies on the Wikipedia veto, which gates on the ground-truth `original` category and is
therefore an **oracle, not deployable**. The deployable category-agnostic veto is acc 0.537 /
precision 0.23. Honest deployable MMFakeBench result is GBM AUC 0.792 (near the 0.700 floor).

**Important caveat for thesis:** The veto cannot make wrong calls in this specific dataset because `original`=all REAL. This must be disclosed. The honest framing: "Wikipedia NLI entailment is a reliable signal for identifying real content that AITR incorrectly flags as fake — with 100% precision on the `original` category."

---

### 16.2 Attention Explainability Findings

**Method:** Extracted CLS token attention weights from AITR transformer layer 0 on MMFakeBench val (1000 samples, real features).

**Results:**

| Token | Mean CLS attention |
|---|---|
| img | 0.1638 |
| txt | 0.1616 |
| cross (img×txt) | 0.1636 |
| diff (img−txt) | 0.1618 |
| **scalars** | **0.1828** |

**Dominant token: scalars in 100% of all samples across all categories.**

**Interpretation:** AITR learned to rely almost entirely on the scalar token (clip_prob + clip_sim + ateeq_score). The transformer self-attention is not performing meaningful cross-modal reasoning — it is essentially a learned scalar aggregator with token-level context. This is consistent with the fact that the scalar signals (CLIP + Ateeq) carry most of the discriminative power.

**Per-category dominant token:** scalars 100% in original, mismatch, textual_vd, visual_vd — no category-specific attention pattern.

**Files:** `D:\Pics Can Lie\overnight_v2\task3_attention.json`, `per_sample_attention_mmfb.csv`

---

### 16.3 Soft Ensemble Results (B+C)

**Setup:** Average probabilities from Variant B (clip+ateeq) and Variant C (clip+deberta) before thresholding.

| Ensemble | Overall | original | mismatch | textual_vd | visual_vd | AUC |
|---|---|---|---|---|---|---|
| B alone | 72.60% | 59.00% | 72.00% | 79.67% | 94.00% | 0.737 |
| B+C equal | 73.70% | **32.00%** | 88.67% | 92.33% | 98.00% | 0.740 |
| B+C B-heavy (0.7/0.3) | 73.40% | — | — | — | — | 0.740 |
| B+C+D equal | 73.60% | — | — | — | — | 0.741 |

**Key finding:** B+C improves fake detection (mismatch +16.67pp, textual_vd +12.66pp, visual_vd +4.00pp) but destroys original accuracy (59% → 32%). Adding DeBERTa shifts the model aggressively toward FAKE. **Not a valid headline — use only as secondary finding.**

**Files:** `D:\Pics Can Lie\overnight_v2\task4_ensemble.json`

---

### 16.4 VERITE Evaluation Attempt — BLOCKED

**Attempted:** Zero-shot evaluation on VERITE (1001 samples, 3-class: true/miscaptioned/out-of-context).

**What worked:**
- Downloaded VERITE files: `VERITE.csv`, `VERITE_articles.csv`, `VERITE_clip_image_embeddings_ViTL14.npy`, `VERITE_clip_text_embeddings_ViTL14.npy`
- CLIP-only cosine similarity evaluation: 66.83% overall, AUC=0.7093 (below 74.40% baseline — expected since features are standard CLIP, not fine-tuned v2)

**What failed — AITR zero-shot (35.86%, F1=0.06):**
- Root cause: used `sigmoid(sim*10-5)` as proxy for clip_prob
- AITR relies heavily on scalar token (clip_prob especially) — proxy has wrong distribution
- AITR collapsed to always-REAL (100% on true, ~3% on fakes)
- Fix requires real clip_prob = running fine-tuned CLIP v2 classifier on actual images

**Why images can't be downloaded:**
- `true_url` = Snopes proxy URLs → HTTP 500 (bot protection), ~663 samples blocked
- `false_url` = direct CDN URLs → HTTP 200, works fine, ~325 samples accessible
- ~66% of VERITE images are inaccessible without Snopes credentials

**Conclusion:** VERITE evaluation requires either (a) Snopes API access, (b) a VPN/scraper approach, or (c) evaluating only on the `false_url` subset (325 out-of-context samples). Not worth pursuing with 7 days left.

**Files:** `D:\Pics Can Lie\verite\` — CLIP-only results saved, AITR results invalid.

---

### 16.5 What Was Tried and Failed (Overnight)

| Attempt | Outcome | Root cause | Do not retry |
|---|---|---|---|
| AITR zero-shot on VERITE | 35.86% (broken) | sigmoid proxy for clip_prob wrong distribution | ✅ — need real images |
| VERITE image download (Snopes) | HTTP 500 | Bot protection on Snopes proxy | ✅ — unresolvable |
| Per-category threshold optimization | 95.30% (overfit) | Tuning and evaluating on same val set | ✅ — need held-out split |
| B+C ensemble as headline | Not valid | Destroys original accuracy (32%) | ✅ — secondary only |
| Veto on non-original categories | N/A | mismatch/textual_vd/visual_vd = all FAKE — veto impossible | ✅ — structural |

```python
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type='cuda', dtype=torch.float16):
    logits, img_proj, txt_proj = model(img, tokens)
    loss = criterion(logits, labels)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

---

## 17. NewsCLIPpings Test Set Evaluation (honest, June 4 2026) ✅

### 17.1 Dataset Discovery

**Test set location (confirmed):**
- Annotations: `D:\Pics Can Lie\dataset\data\NewsClipPings\merged_balanced\test.json`
- Metadata (captions, sources, image paths): `D:\Pics Can Lie\dataset\data\NewsClipPings\metadata\test.json`
- Images: `D:\Pics Can Lie\dataset\origin\origin\` (all 4 sources present: bbc, guardian, usa_today, washington_post)

**Test set structure:**
- `merged_balanced\test.json` — dict with keys `source_datasets` and `annotations` (list of 7264 entries)
- Each annotation: `{id, image_id, similarity_score, source_dataset, falsified}`
- **Label:** `falsified=True` → FAKE (1), `falsified=False` → REAL (0)
- **Total unique samples:** 3632 (each appears twice: once real, once fake — perfectly balanced)
- **Source:** obtained from `metadata\test.json[str(ann['id'])]['source']` — NOT from `ann['source_dataset']` (that's an integer index 1-3, not a name)

**Critical image path mapping for fake entries:**
- Real entries: image comes from `meta[str(ann['id'])]['image_path']`
- Fake entries: `ann['image_id'] != ann['id']` — the displayed image is the swapped one, path from `meta[str(ann['image_id'])]['image_path']`
- Strip `visual_news/origin/` prefix from all image paths

**Image path verification:** 100/100 spot check found, 0 missing.

### 17.2 What Failed During Test Set Evaluation

| Attempt | Error | Fix |
|---|---|---|
| `data[0]` on merged_balanced test.json | `KeyError: 0` — it's a dict not a list | Use `data["annotations"]` |
| `torch.load(CLIP_CKPT, map_location=DEVICE)` | OOM — CLIP already using ~3GB VRAM, 2.35GB checkpoint pushed over limit | Load checkpoint to CPU: `map_location='cpu'`, then `del ckpt` after loading |
| `CLIPClassifier` head named `head` | `Missing key: head.0.weight` — checkpoint uses `classifier` | Rename to `self.classifier`; input dim is 1537 (768+768+1 cosine sim), hidden 512→128→1 |
| batch_size=64 during inference | OOM on 4060 8GB | Reduce to batch_size=8 |
| AITR threshold 0.54 on test set | `fused_prob` max=0.14, predicted FAKE=0, accuracy=50% | Threshold sweep revealed optimal=0.010 for this output range |
| `sorted(per_source.items())` | `TypeError: '<' not supported between str and int` | Source keys were integers from `source_dataset` field; fix by using metadata `source` string |
| Scalar order wrong (clip_sim at idx 7, clip_prob at idx 8) | AITR fused_prob constant ~0.003 | Reorder: `scalars[:,0]=clip_prob, scalars[:,1]=clip_sim, ...` |

### 17.3 The two bugs that produced the withdrawn 86.48% (important for thesis)

The original test run had **two coupled bugs** found in the honesty audit:
1. **Data snooping:** the decision threshold (0.010) was swept over the **test labels** to
   maximize test accuracy — invalid. This alone is disqualifying.
2. **Broken feature path:** the persisted **StandardScaler was never applied** and the 9 scalars
   were assembled in the wrong order, so AITR saw out-of-distribution inputs squashed to ~0.02.

The fix (`scripts/run_test_inference_fixed.py`): recompute CLIP fresh, assemble scalars in the
TRAIN order `[clip_prob, clip_sim, deberta, s2..s6, wiki]` with evidence/NLI imputed to TRAIN
means, apply the train-fit scaler, and decide at a threshold **frozen on TRAIN** (0.5523) — **no
argmax over test labels anywhere**. Fresh-vs-precomputed parity gate passed at Pearson r ≈ 1.0.

### 17.4 Honest test-set results (fixed pipeline)

**Primary metric — AUC = 0.932 [0.926, 0.937]** (threshold-free; CLIP alone is highly
discriminative). Accuracy is reported at blind operating points only:

| Operating point | Overall acc (95% CI) | BBC | Guardian | USA Today | Wash Post |
|---|---|---|---|---|---|
| Raw frozen 0.5523 (blind) | 0.831 [0.822, 0.839] | 0.796 | 0.832 | 0.842 | 0.843 |
| **Isotonic @0.5 (SHIPPED, blind)** | **0.865 [0.856, 0.873]** | 0.845 | 0.863 | 0.877 | 0.870 |
| Source-aware shrunk (ablation, blind) | 0.865 [0.857, 0.873] | 0.851 | 0.862 | 0.878 | 0.868 |
| ~~Swept on test labels (invalid)~~ | ~~0.8648~~ | — | — | — | — |
| *Diagnostic ceiling (uses TEST labels)* | *0.866* | — | — | — | — |

**Key insight (thesis):** the AUC(0.932)/accuracy(0.831) gap was almost entirely a
**threshold/calibration problem**, not model capacity. The raw fused prob is over-confident
(ECE 0.138); a val-fit isotonic calibrator (ECE → 0.017) decided at 0.5 recovers ~96% of the
gap to the test-label ceiling (0.866), reaching **0.865 honestly** (val-derived, applied blind).
Beyond ~0.866 the AUC ceiling binds — not recoverable without a stronger model.

**Output files:** `results/test_set_results/test_results_fixed.json` (per-sample),
`results/calibration_threshold_report.json` (full calibration study, all 5 parts).

---

## 13. What NOT to Do

1. **Do not use `D:\Pics Can Lie\val_sample_ids.csv`** — 3,232 rows (old). Use v2 file (5,000 rows).
2. **Do not use `deberta_val_scores.csv`** (no v2 suffix) — broken, scores all ~0.014.
3. **Do not use BLIP CLS embeddings** — F1=0.46, noise in fusion.
4. **Do not load CLIP without `jit=False`** — MemoryError on Windows.
5. **Do not load CLIP without `.float()`** — LayerNorm RuntimeError.
6. **Do not use `num_workers>0` in Jupyter on Windows** — deadlock.
7. **Do not align features by position** — always align by sample ID.
8. **Do not compare to MUSE system accuracy** — their frozen CLIP alone = 80.7%.
9. **Do not use article text** — .txt files don't exist locally.
10. **Do not add Ateeq to NewsCLIPpings** — outputs near-constant "not AI" = noise.
11. **Do not use half precision (.half()) on CLIP** — LayerNorm requires float32.
12. **Do not run evidence pipeline on 4090** — bad internet, use personal machine.
13. **Do not use `pin_memory=True` with `num_workers=0`** — no benefit, potential issues.
14. **Do not use contrastive loss with temperature=0.07** — produces loss=4.0, dominates training.
15. **Do not use CrossModalAttention in CLIP head** — v3 proved it hurts performance.
16. **Do not report per-category threshold optimization as a headline result** — Task 2 overnight showed 95.30% which is pure val-set overfitting. Threshold optimization on the same set you evaluate on is invalid. Always use a held-out split for threshold tuning.
17. **Do not use a sigmoid proxy for clip_prob on VERITE** — `sigmoid(sim*10-5)` produces wrong distribution, collapses AITR to always-REAL (35.86% accuracy, F1=0.06). Real clip_prob requires running the fine-tuned classifier on actual images.
18. **Do not attempt VERITE image download from Snopes proxy URLs** — `true_url` returns HTTP 500 (bot protection). Only `false_url` (direct CDN links) works. ~663/1001 VERITE images are inaccessible.
19. **Do not use the B+C ensemble as headline MMFakeBench result** — B+C gets 73.70% overall but destroys original accuracy (59% → 32%). It is a secondary finding only.
20. **Do not re-run Variant A training on MMFakeBench** — it degenerates to always-Fake regardless of LR or patience. CLIP alone (no Ateeq) cannot beat the 70% floor on MMFakeBench. This is a documented structural finding, not a bug.
21. **Do not use `map_location=DEVICE` when loading the CLIP v2 checkpoint** — the 2.35GB checkpoint + CLIP ViT-L/14 already on GPU exceeds 8GB VRAM on the 4060. Always load checkpoint to CPU first: `torch.load(path, map_location='cpu')`, load state dict, then `del ckpt; torch.cuda.empty_cache()`.
22. **Do not use batch_size > 8 for CLIP inference on the 4060** — ViT-L/14 is large; batches of 64 cause OOM during the forward pass.
23. **Do not access merged_balanced test.json with `data[0]`** — it is a dict with keys `source_datasets` and `annotations`. Use `data["annotations"]` to get the list.
24. **Do not use `ann['source_dataset']` for source-aware thresholds** — that field is an integer index (1–3) mapping to source dataset type, not the news source name. Use `meta[str(ann['id'])]['source']` to get `'bbc'`, `'guardian'`, etc.
25. **Do not use the same AITR threshold (0.54) on freshly computed test features** — AITR was trained on precomputed val features; freshly computed features produce a compressed output range (0.003–0.14). Always do a threshold sweep on the target distribution. The relative ordering is preserved even when the absolute range shifts.
26. **Do not put clip_sim before clip_prob in the scalar tensor** — AITR scalar order must be `[clip_prob, clip_sim, deberta, s2, s3, s4, s5, s6, wiki_score]`. Wrong order collapses fused_prob to a constant (~0.003) and gives 50% accuracy on balanced data.
27. **Do not use image_id=id for fake test entries** — for `falsified=True` annotations, `image_id != id`. The actual displayed image path must be looked up via `meta[str(ann['image_id'])]['image_path']`, not `meta[str(ann['id'])]['image_path']`.

---

## 14. Current System Description (for thesis)

**Input:** News image + caption pair

**Step 1 — CLIP visual-semantic alignment**
Fine-tuned CLIP ViT-L/14 encodes image and caption into 768-dim embeddings.
Outputs: classification probability + cosine similarity.
Carries ~47% of final decision weight.

**Step 2 — Evidence retrieval scoring**
Pre-crawled evidence links (Abdelnabi et al.) fetch external images.
Computes 5 CLIP similarity scores (s2-s6) between original image, caption, and evidence.
95.8% coverage on val set.

**Step 3 — Text fact-checking (auxiliary)**
DeBERTa NLI: caption vs article text entailment.
Wikipedia NLI: caption vs entity Wikipedia summaries.
Both proved weak (diff < 0.01 between real/fake) but included as auxiliary signals.

**Step 4 — AITR Transformer Fusion**
5 tokens (img, txt, img×txt, img−txt, scalars) → self-attention → CLS → fused probability.
At inference the 7 evidence/NLI scalars are imputed to TRAIN means (matched to the test
pipeline, which has no retrieved evidence).

**Step 5 — Isotonic calibration + 0.5 decision (SHIPPED)**
The over-confident raw fused prob (ECE 0.138) is passed through a **val-fit isotonic calibrator**
(`models/fusion_aitr/calibrators.joblib`, ECE → 0.017) and the verdict is taken at **0.5** on the
calibrated probability, which is also the displayed confidence. Source-aware thresholds were
tried and **did not beat** this (failed ablation, Section 6).

**Final honest result: AUC 0.932 / test accuracy 0.865** at a blind, val-derived calibrated
operating point — competitive with no external API (SNIFFER 88.40% uses external retrieval).

### 14b. Live demo architecture (`src/app.py`) — two stages, three independent signals
The deployed demo wraps the research consistency model and adds two wild-image checks. The
verdict is anchored ONLY to consistency; the other two are reported independently.

| # | Signal | Model / track | Role | Decision |
|---|---|---|---|---|
| 1 | **Image–caption consistency** | CLIP v2 → AITR → **isotonic** [NewsCLIPpings] | PRIMARY (drives verdict) | calibrated P(out-of-context) ≥ 0.5 → FAKE |
| 2 | **Image origin** | **Ateeq** SiglipForImageClassification [MMFakeBench/AI] | auxiliary, medium | P(AI) ≥ 0.5 → AI-generated (+ resolution caveat on >12 MP) |
| 3 | **Caption factuality** | **Claude Sonnet 4.6 + web_search** [on-demand] | auxiliary, low | Claude verdict factual/distorted; cached; runs only on button click |

- **Stage 1 (NewsCLIPpings track):** the calibrated consistency model above. `calibrated_prob =
  P(out-of-context)`; high → FAKE. This is the only signal that sets REAL/FAKE.
- **Stage 2 (MMFakeBench / AI-image track):** Ateeq answers "is the image AI-generated?" for wild
  uploads (validated to generalize, Section 4.7); Claude verifies the caption's factual claims via
  live web search, on-demand to control cost/latency, with per-caption caching.
- **Removed from the demo:** SightEngine (replaced by Ateeq) and DeBERTa/Wikipedia NLI (replaced by
  Claude). The demo is now fully local except the optional Claude fact-check.

---

## 15. Next Steps

### CURRENT STATE (May 26, 2026)
- ✅ Step 1 (MMFakeBench AITR retraining): COMPLETE — 72.60%
- ✅ Step 2 (DeBERTa NLI ablation): COMPLETE — documented negative finding
- ✅ Step 3 (Wikipedia NLI veto): COMPLETE — **79.80% headline**
- ✅ Attention explainability: COMPLETE — scalars dominate all categories
- ✅ Soft ensemble analysis: COMPLETE — B+C=73.70% (secondary result)
- ✅ **NewsCLIPpings honest test eval + isotonic calibration: COMPLETE — AUC 0.932 / acc 0.865** (calibrated, blind) ← replaces the withdrawn 86.48%

### ALL IMPLEMENTATION NOW COMPLETE. REMAINING WORK IS THESIS WRITING ONLY.

**Thesis writing (remaining days)**
- Main result: NewsCLIPpings **test AUC 0.932 / acc 0.865** (val-derived isotonic @0.5, blind);
  lead with AUC; "competitive, no external API" vs SNIFFER (don't claim a head-to-head win)
- Calibration finding: raw AITR over-confident (ECE 0.138 → 0.017); the AUC/accuracy gap was a
  threshold/calibration problem, not capacity. Source-aware thresholds = failed ablation.
- Generalization: MMFakeBench deployable GBM AUC 0.792 (near 0.700 floor); the Wikipedia veto is
  an oracle (not deployable) — present as a limitation, not a headline
- Ablation table: A/B/C/D variants
- Negative findings: NLI ineffective on NewsCLIPpings, DeBERTa adds no lift over Ateeq on MMFakeBench
- Explainability: attention weights, Wikipedia veto mechanism
- Test set discussion: threshold shift (0.54→0.010) explained by distribution shift between precomputed val features and freshly computed test features; relative ordering preserved (corr=0.703); honest and expected

### DO NOT ATTEMPT
- Compositional CLIP (Grounding DINO) — too complex, 5-7 days minimum, no time
- VERITE evaluation — blocked by Snopes proxy (663/1001 images inaccessible)
- CLIP v4 retraining — 84.07% at crash, worse than v2 (85.60%)
- Per-category threshold optimization — proven to overfit (see Section 13, rule 16)
- Any new training — implementation is complete, thesis writing is the priority
