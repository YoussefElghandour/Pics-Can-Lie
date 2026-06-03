# Overnight Run - DONE
Date: 2026-05-25 06:14

---

## Task 1 - Data Verification
PASSED - 1001 samples, shapes (1001,768) confirmed, no NaN/Inf.
Labels: true=338, miscaptioned=338, out-of-context=325

## Task 2 - CLIP-only Results
Best threshold: 0.3
Overall accuracy: 66.83%
AUC-ROC: 0.7093  |  F1: 0.7426
  true (REAL):           56.21%
  miscaptioned (FAKE):   56.80%
  out-of-context (FAKE): 88.31%

## Task 3 - AITR Fusion Results
Best threshold: 0.1
Overall accuracy: 35.86%
At thr=0.54 (NewsCLIPpings optimal): 33.77%
AUC-ROC: 0.7053  |  F1: 0.0614
  true (REAL):           100.00%
  miscaptioned (FAKE):   0.30%
  out-of-context (FAKE): 6.15%

## Task 4 - Attention Explainability
Mean CLS attention weights:
  img: 0.1636
  txt: 0.1609
  cross: 0.1636
  diff: 0.1609
  scalars: 0.1851
Dominant token distribution:
  img: 0.0%
  txt: 0.0%
  cross: 0.0%
  diff: 0.0%
  scalars: 100.0%

## Task 5 - Error Analysis
Total errors: 642 / 1001
FP (real->fake): 0
FN (fake->real): 642
  FN miscaptioned: 337
  FN out-of-context: 305
CLIP sim correct: 0.2948  |  incorrect: 0.2707

## Task 6 - Comparison Table
  CLIP ViT-L/14 zero-shot (VERITE paper baseline): 74.40%
  VERITE paper best (trained on VERITE): 81.00%
  LAMAR (new SOTA Apr 2025): TBD
  MAD-Sherlock (ICML 2025): TBD
  Your CLIP-only zero-shot: 66.83% (AUC=0.7093)
  Your AITR fusion zero-shot: 35.86% (AUC=0.7053)

## Files Written
- clip_only_results.json
- aitr_fusion_results.json
- attention_analysis.json
- per_sample_attention.csv
- error_analysis.json
- comparison_table.json
- task1_verify.py through task7_done.py

## Key Questions to Answer in the Morning
1. Does AITR beat CLIP-only? (does fusion generalize to VERITE?)
2. Is overall accuracy above 74.40%? (zero-shot generalization win)
3. Which token dominates attention? (explainability finding)
4. Are miscaptioned errors higher than OOC? (expected - NLI weakness)
5. Does thr=0.54 transfer or does VERITE need its own threshold?