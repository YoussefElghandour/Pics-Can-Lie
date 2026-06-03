# Pics Can Lie — Complete Thesis Writing Notes (Updated April 14, 2026)

---

## PROJECT OVERVIEW

**Project name:** Pics Can Lie
**Task:** Multimodal misinformation detection — detecting out-of-context images
**Location:** D:\Pics Can Lie
**Stack:** Python 3.10, PyTorch, HuggingFace Transformers, Gradio, Ollama
**Datasets:** NewsClipPings (primary), MMFakeBench (secondary/module evaluation)

---

## PIPELINE ARCHITECTURE

Five modules:
1. BLIP ITM (fine-tuned) — image-text matching, detects visual mismatch
2. DeBERTa NLI — article vs caption consistency checking
3. Fine-tuned AI Detection (Ateeqq/ai-vs-human-image-detector, SiglipForImageClassification) — visual authenticity
4. Logistic Regression Fusion — combines all module scores
5. LLaMA 3.1 8B (via Ollama) — natural language explanation (VRAM issues, not fully resolved)

---

## COMPLETED RESULTS

### 1. BLIP ITM Fine-Tuning (NewsClipPings)
- Pretrained zero-shot: ~62% accuracy
- Fine-tuned epoch 9: 74.9% accuracy, 75.1% F1
- Training: 50k balanced samples, 9 epochs, LR 1e-5, batch 16, early stopping patience 3
- Checkpoint saved to: blip_itm_finetuned/

**Thesis paragraph:**
"Fine-tuning the BLIP ITM model on the NewsClipPings training set yielded a substantial 
improvement over the zero-shot pretrained baseline. Starting from approximately 62% accuracy, 
the fine-tuned checkpoint achieved 74.9% accuracy and 75.1% F1 after 9 epochs of training 
with early stopping. This 13-point gain demonstrates that domain-specific adaptation is 
critical for out-of-context detection, where general vision-language pretraining on web data 
does not adequately capture the subtle mismatches present in news image-caption pairs."

---

### 2. DeBERTa NLI Standalone (NewsClipPings, n=1000)
- Accuracy: 60.2%, F1: 70.6% (inflated — class biased)
- Optimal threshold: 0.65
- Confusion matrix: 479/500 fakes caught, 377/500 real misclassified as fake
- Real class accuracy: only 24.6%

**Thesis paragraph:**
"Evaluated as a standalone classifier, the DeBERTa NLI module achieved 60.2% accuracy and 
a misleadingly high F1 of 70.6%. Inspection of the confusion matrix reveals that this F1 
score is inflated by a strong class bias: the module correctly flagged 479 of 500 fake 
samples (95.8% recall on fake) but misclassified 377 of 500 real samples as fake, yielding 
only 24.6% accuracy on the real class. The threshold sweep from 0.3 to 0.7 produced 
negligible changes in accuracy (57.6% to 60.2%), confirming that the module lacks a 
meaningful decision boundary. This behavior is explained by the nature of NewsClipPings: 
since both image and caption are authentic, the article text legitimately entails the caption 
in both real and fake samples — making NLI-based internal consistency checking insufficient 
as a standalone signal. However, as a fusion feature, DeBERTa contributes a complementary 
signal that improves overall fusion accuracy by 2.3 percentage points over BLIP alone."

---

### 3. Fusion Ablation (NewsClipPings, n=300, all on same samples — fair comparison)
- BLIP only: 82.67% accuracy, 82.56% F1
- BLIP + DeBERTa: 85.0% accuracy, 84.84% F1 (+2.3pp)
- BLIP + SightEngine: 88.0% accuracy, 88.09% F1 (+5.4pp)
- BLIP + DeBERTa + SightEngine (full): 89.67% accuracy, 89.58% F1 (+7.0pp)
- Note: SightEngine limited to 300 samples due to API quota

**Thesis paragraph:**
"The ablation study reveals that fine-tuned BLIP alone matches the pretrained 3-module 
baseline of 82.67%, establishing fine-tuning as the dominant contributor to performance 
gains. Adding DeBERTa as a second signal yields a 2.3 percentage point improvement (85.0%), 
while adding SightEngine instead produces a larger gain of 5.4 points (88.0%). The full 
3-module fusion achieves the best result at 89.67% accuracy and 89.58% F1, representing 
a 7-point improvement over BLIP alone. The stronger contribution of SightEngine relative 
to DeBERTa is notable given that NewsClipPings contains exclusively real, unedited images 
— suggesting that SightEngine's signal functions as a proxy for low-level image statistics 
such as compression artifacts and sensor noise patterns, rather than as a true AI-generation 
detector."

---

### 4. AI Detection Module — Pretrained Evaluation (MMFakeBench val, n=200)
- Overall accuracy: 56.0%, F1: 52.2%
- PS-edited detection: 12/50 = 24%
- AI-generated detection: 72% (initial eval), ~100% on clean DALL-E images
- FP rate on VisualNews real photos: 34/71 = 47.9%
- FP rate on MS-COCO real: 1/17 = 5.9%
- FP rate on Fakeddit real: 1/12 = 8.3%
- Raw NewsClipPings FP rate (unprocessed): 67-68%

**Thesis paragraph:**
"We evaluate the AI detection module (Ateeqq/ai-vs-human-image-detector, 
SiglipForImageClassification) on MMFakeBench's visual veracity distortion subset. 
The model achieves 56.0% overall accuracy and 52.2% F1 — barely above random chance 
— driven by two failure modes. First, the model detects only 24% of Photoshop-edited 
images (12/50), as these retain real photographic pixel statistics that the model was 
not trained to flag. Second, the model produces a 47.9% false positive rate on VisualNews 
real news photographs, misclassifying nearly half as AI-generated. In contrast, MS-COCO 
and Fakeddit real images produce near-zero false positives (5.9% and 8.3% respectively), 
indicating strong domain sensitivity to the compression and post-processing characteristics 
of professional news photography. This finding is consistent with the 'Fact or Fake' study 
(2026), which demonstrated that pixel-level detectors introduce misleading authenticity 
priors in news verification pipelines. Accordingly, we use the raw AI score as a continuous 
fusion feature rather than a binary classifier, allowing the logistic regression meta-learner 
to assign appropriate weight to this signal."

---

### 5. AI Detection Module — Fine-Tuning (Domain Adaptation)
**Training data:**
- Real: 500 NewsClipPings origin images (seed=42)
- Fake: 500 MMFakeBench TEST set images (250 AI-generated + 250 PS-edited)
- Total: 1000 samples, 800 train / 200 val, stratified
- Training: 6 epochs, 2-phase (classifier only → + last 2 encoder layers)
- Time: ~2 minutes on CUDA

**Results — Before vs After (held-out MMFakeBench val, n=200):**

| Metric | Before | After | Delta |
|---|---|---|---|
| Overall accuracy | 56.0% | 75.5% | +19.5pp |
| Overall F1 | 52.2% | 78.0% | +25.9pp |
| FP rate on VisualNews | 47.9% | 29.6% | -18.3pp |
| PS-edited detection | 24.0% | 84.0% | +60.0pp |
| AI-generated detection | 72.0% | 90.0% | +18.0pp |

**Overfitting verification — FP rate across 3 independent test sets:**

| Test Set | Before | After |
|---|---|---|
| MMFakeBench val VisualNews (held-out) | 47.9% | 29.6% |
| New NewsClipPings sample (seed=99, 200 imgs) | 67.0% | 27.0% |
| NewsClipPings val.json real (100 imgs) | 68.0% | 16.0% |

- Val loss fell every epoch, no divergence — clean convergence confirmed
- Generalization confirmed across all 3 independent populations

**Thesis paragraph:**
"Domain-specific fine-tuning of the visual authenticity module produced substantial 
improvements across all evaluation categories. Training on 500 real VisualNews photographs 
and 500 MMFakeBench manipulation samples (250 AI-generated, 250 Photoshop-edited) for 6 
epochs yielded a 19.5 percentage point accuracy gain and a 25.9 point F1 improvement on 
the held-out evaluation set. Most notably, Photoshop-edited image detection improved from 
24% to 84% — a 60 point gain demonstrating that the pretrained model's blind spot for 
photo manipulation was directly addressable through targeted training data. The false 
positive rate on real VisualNews photographs decreased from 47.9% to 29.6%, confirming 
that domain adaptation to news photography characteristics reduces the authenticity prior 
mismatch identified in the pretrained model. Crucially, no metric regressed — the 
fine-tuning improved all subgroups simultaneously, indicating that the base 
SiglipForImageClassification architecture has sufficient capacity to jointly model news 
photography, AI generation, and Photoshop manipulation."

"To verify generalization, the fine-tuned model was evaluated on three independent test 
sets with zero training overlap. The false positive rate on real news photographs decreased 
consistently across all three populations: from 47.9% to 29.6% on the MMFakeBench 
validation set, from 67.0% to 27.0% on a new NewsClipPings sample (seed=99), and from 
68.0% to 16.0% on the official NewsClipPings validation split. The training history shows 
val loss decreasing monotonically across all 6 epochs with no divergence between train and 
val accuracy, confirming clean convergence with no memorization. Notably, the pretrained 
model's true false positive rate on unprocessed news photographs (67-68%) was substantially 
higher than the curated MMFakeBench baseline suggested (47.9%), indicating that prior 
evaluation on normalized images underestimated the domain sensitivity problem."

---

### 6. Fine-Tuned AI Detector — Standalone Fusion Comparison (5-fold CV)
| Module | Accuracy | F1 | Std |
|---|---|---|---|
| Pretrained AI detector | 56.5% | 52.7% | ±9.6% |
| Fine-tuned AI detector | 80.0% | 80.1% | ±5.0% |

**Thesis paragraph:**
"As a standalone classifier on MMFakeBench's visual veracity distortion subset, the 
fine-tuned visual authenticity module achieves 80.0% accuracy and 80.1% F1 (±5.0%), 
compared to 56.5% accuracy and 52.7% F1 (±9.6%) for the pretrained baseline — a 23.5 
percentage point improvement. The reduction in standard deviation across cross-validation 
folds (from ±9.6% to ±5.0%) further demonstrates that domain-specific fine-tuning improves 
not only accuracy but also prediction stability, addressing the reliability concerns 
associated with applying static pretrained detectors to news domain images."

---

## DATASET COMPARISON

| Dimension | NewsClipPings | MMFakeBench |
|---|---|---|
| Total samples | ~85,000 | ~11,000 |
| Train/Val/Test | 71k/7k/7k | 6k/1k/10k |
| Text manipulated? | Never | 30% of samples |
| Image manipulated? | Never | 10% PS-edited |
| AI-generated images? | Never | 10% AI-generated |
| Image-text mismatch? | Always (the misinformation) | 30% of samples |
| Fact-checking useful? | No — captions factually true | Yes |
| AI detection useful? | Marginal (proxy signal) | Yes — direct signal |
| BLIP ITM useful? | Primary signal | Useful for OOC subset |
| Published SOTA F1 | ~90% (LAMAR) | ~81% (MIRAGE) |
| Human accuracy | ~66% | ~56% |

**Thesis paragraph:**
"NewsClipPings and MMFakeBench are complementary benchmarks that stress-test fundamentally 
different aspects of multimodal misinformation. NewsClipPings contains exclusively authentic, 
unedited images and captions — misinformation arises purely from the mismatch between a 
real image and a real caption from a different context. This makes image-text consistency 
(BLIP ITM) and internal article-caption NLI (DeBERTa) the primary useful signals, while 
AI detection and fact-checking are uninformative since captions are factually true. 
MMFakeBench, by contrast, contains textually false claims (30%), AI-generated images (10%), 
and Photoshop-edited images (10%), alongside out-of-context pairs (30%) — making 
fact-checking and visual authenticity modules meaningful for the first time. Evaluating 
our pipeline on both benchmarks allows us to isolate each module's contribution to different 
misinformation types and demonstrate the generalizability of the modular fusion approach."

---

## LITERATURE-SUPPORTED ARGUMENTS

**Against detector fusion (cite: "Fact or Fake", Feb 2026):**
"A February 2026 study systematically evaluated the role of deepfake detectors in 
multimodal misinformation pipelines on MMFakeBench and DGM4. Standalone detectors achieved 
F1 scores of only 0.26–0.53 on MMFakeBench, and integrating detector outputs into an 
evidence-centric fact-checking pipeline consistently reduced F1 by 0.04–0.08 due to 
non-causal authenticity assumptions. Our empirical results on MMFakeBench's visual veracity 
distortion subset corroborate this finding: the pretrained AI detection module achieves only 
56% accuracy overall, with a 47.9% false positive rate on real news photographs — confirming 
that authenticity priors learned from synthetic image datasets do not transfer reliably to 
news domain verification tasks. Domain-specific fine-tuning addresses this limitation."

**Modular fusion vs LLM approaches:**
"Recent state-of-the-art systems such as MIRAGE (2025) and MMD-Agent achieve competitive 
performance on MMFakeBench through a single large vision-language model prompted with 
decomposed subtasks and augmented with web retrieval, achieving 74–81% F1. While powerful, 
these systems rely on paid proprietary APIs, exhibit stochastic output that limits 
reproducibility, and provide no interpretable numeric signal per module. Our modular fusion 
approach offers a locally deployable, fully reproducible alternative with explicit per-module 
scores that enable direct ablation analysis — a deliberate design choice that prioritizes 
interpretability and scientific rigor over raw benchmark performance."

---

## COMPLETE RESULTS SUMMARY TABLE

| System / Configuration | Dataset | Accuracy | F1 | Notes |
|---|---|---|---|---|
| BLIP ITM pretrained zero-shot | NewsClipPings | ~62% | ~60% | Baseline |
| BLIP ITM fine-tuned (epoch 9) | NewsClipPings | 74.9% | 75.1% | +13pp gain |
| DeBERTa standalone | NewsClipPings | 60.2% | 70.6%* | *inflated by bias |
| BLIP + DeBERTa | NewsClipPings | 85.0% | 84.84% | n=300 |
| BLIP + SightEngine | NewsClipPings | 88.0% | 88.09% | n=300 |
| BLIP + DeBERTa + SightEngine | NewsClipPings | 89.67% | 89.58% | n=300, best |
| Pretrained AI detector standalone | MMFakeBench | 56.5% | 52.7% | ±9.6% std |
| Fine-tuned AI detector standalone | MMFakeBench | 80.0% | 80.1% | ±5.0% std |
| MIRAGE (SOTA, GPT-4o-mini) | MMFakeBench | 75.1% | 81.65% | Published SOTA |
| MMD-Agent (GPT-4V) | MMFakeBench | 76.8% | 74.0% | Published baseline |

---

## WHAT REMAINS TO DO

### Must Do (thesis critical):
1. **Wikipedia fact-checking module** — implement on MMFakeBench
   - Approach: spaCy NER → Wikipedia API → DeBERTa entailment
   - Only meaningful on MMFakeBench (NewsClipPings captions are factually true)
   - This is the new contribution the supervisor requested

2. **Update Gradio demo** — replace pretrained AI detector with fine-tuned model
   - Fix LLaMA VRAM issue or switch to lighter model (3B or CPU)

3. **Write the thesis** — all numbers and paragraphs are ready in this document

### Nice to Have (if time allows):
4. Scale fusion to 1000 samples on NewsClipPings
5. BLIP-2 fine-tuning on Kaggle (was blocked by hardware)
6. Full fusion on MMFakeBench (decided not necessary — module-level eval is sufficient)

---

## HANDOFF NOTE FOR NEW CHAT

Copy this into any new Claude chat to restore full context:

---
"I am working on my thesis project called 'Pics Can Lie' — a multimodal misinformation 
detection system for detecting out-of-context news images.

Project location: D:\Pics Can Lie
Primary dataset: NewsClipPings (85k samples, BBC/Guardian/USA Today/Washington Post)
Secondary dataset: MMFakeBench (11k samples, mixed misinformation types, downloaded)

Completed modules:
- BLIP ITM fine-tuned: 74.9% accuracy on NewsClipPings
- DeBERTa NLI: 60.2% standalone (biased but contributes to fusion)
- Full fusion (BLIP + DeBERTa + SightEngine): 89.67% on NewsClipPings
- AI detection fine-tuned (Ateeqq SiglipForImageClassification): 80.0% on MMFakeBench
- Fine-tuned model saved at: D:\Pics Can Lie\ai_detector_finetuned\
- All scores cached in CSV files in project root

Next task: Implement Wikipedia fact-checking module
- Use spaCy for named entity extraction from captions
- Query Wikipedia API for each entity
- Run DeBERTa NLI between Wikipedia text and caption
- Return entailment score as fact-checking signal
- Evaluate on MMFakeBench val set (captions contain false claims)
- Save scores to: D:\Pics Can Lie\mmfakebench_factcheck_scores.csv"
---

