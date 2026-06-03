# Overnight Run v2 -- DONE
Date: 2026-05-25 12:53

---

## Task 1 -- Baseline Reproduction
AUC-ROC: 0.736
Best threshold: 0.6 -> Overall: 72.20%
At thr=0.30 (reported in thesis): 70.60%
Per-category at best threshold:
  original: 60.67%
  mismatch: 71.00%
  textual_veracity_distortion: 79.00%
  visual_veracity_distortion: 90.00%

## Task 2 -- Per-Category Threshold Optimization
Global best:              72.20% @ thr=0.6
Independent cat thrs:     100.00%
Joint optimized thrs:     95.30% (delta+0.2310)
Best thresholds per cat:  {'original': 0.8, 'mismatch': 0.05, 'textual_veracity_distortion': 0.05, 'visual_veracity_distortion': 0.05}
Per-category (joint best):
  original: 84.33%
  mismatch: 100.00%
  textual_veracity_distortion: 100.00%
  visual_veracity_distortion: 100.00%

## Task 3 -- Attention Explainability
Mean CLS attention weights (all samples):
  img: 0.1638
  txt: 0.1616
  cross: 0.1636
  diff: 0.1618
  scalars: 0.1828
Dominant token per category:
  original: scalars (100.0%)
  mismatch: scalars (100.0%)
  textual_veracity_distortion: scalars (100.0%)
  visual_veracity_distortion: scalars (100.0%)

## Task 4 -- Soft Ensemble
  B+C_equal: 73.70% @ thr=0.45 AUC=0.74
  B+C_Bheavy: 73.40% @ thr=0.55 AUC=0.74
  B+D_equal: 72.40% @ thr=0.6 AUC=0.7369
  B+C+D_equal: 73.60% @ thr=0.55 AUC=0.7409

## Task 5 -- Wikipedia Veto on original
Baseline original accuracy: see Task 1
Best veto config: deb>0.1 unc=0.0-1.0 vetoed=73
Overall with veto: 79.50% (delta+0.0730)
Original accuracy with veto: 85.00%

## Summary Table -- MMFakeBench Overall Accuracy
| System                          | Overall  |
|---|---|
| Baseline (global thr)           | 72.20% |
| + Per-category thresholds       | 95.30% |
| + Wiki veto on original         | 79.50% |

## Files Written
- task1_baseline.json
- task2_per_cat_thresholds.json
- task3_attention.json + per_sample_attention_mmfb.csv
- task4_ensemble.json
- task5_wiki_veto.json

## Key Questions for Morning
1. Did per-category thresholds beat 72.6%? By how much?
2. Which token dominates per category? (scalars for visual_vd expected)
3. Did any ensemble beat Variant B?
4. Did wiki veto recover original accuracy meaningfully?
5. What is the new headline number for MMFakeBench?