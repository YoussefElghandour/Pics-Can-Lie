# Pics Can Lie — Corrected Results & Claims (thesis scaffold)

Single source of truth for the Results and Discussion chapters. Every number is
from the honest, audited pipeline (frozen thresholds, AUC-primary, bootstrap CIs).
Derived from `results/*.json`; full comparison in `docs/comparison_table.md`.

---

## 1. Honest results table

### NewsCLIPpings (in-domain) — CLIP v2 → AITR
| Split | AUC (95% CI) | Accuracy (95% CI) | Operating point |
|---|---|---|---|
| VAL  | 0.9477 [0.9351, 0.9605] | 0.880 [0.861, 0.900] | 0.5523 (train-frozen) |
| TEST (production) | **0.9317 [0.9258, 0.9371]** | **0.865 [0.856, 0.873]** | **isotonic @0.5** (val-derived, blind) |
| TEST (old, superseded) | 0.9317 [0.9258, 0.9371] | 0.8308 [0.8221, 0.8392] | 0.5523 raw frozen |

**Production operating point** is the val-fit isotonic calibrator decided at 0.5 (not the
raw-prob frozen 0.5523). **AUC is unchanged at 0.932** — calibration is monotonic and does
not alter ranking. Accuracy improves 0.831 → **0.865** (CIs barely overlap), recovering ~96%
of the gap to the test-label thresholding ceiling (0.866). Invalid prior headline (threshold
swept on test, 0.8648) remains retired.

**Calibration finding.** The raw AITR fused prob is badly **over-confident**: mean predicted
confidence 0.96 vs accuracy 0.83, mass piled at 0/1, **ECE 0.138** on test. The frozen 0.5523
was mismatched to this distribution (the true optimum sits near 0.02 because the imputed
evidence/NLI scalars depress fused probs). A **val-fit isotonic** calibrator cuts **ECE 0.138 →
0.017** and lets the verdict be taken at a clean, transferable 0.5. Platt scaling also helps
(ECE → 0.054, acc 0.856) but isotonic is better. This was a **threshold/calibration problem,
not a model-capacity problem** — beyond ~0.866 the AUC ceiling binds and is not honestly
recoverable without a stronger model.

**Source-aware thresholds — FAILED ABLATION (not shipped).** Per-source thresholds derived on
val and shrunk toward the global threshold (pseudo-count k=1000) scored 0.8652 [0.857, 0.873]
on test — **statistically indistinguishable** from global isotonic (0.8647). The raw per-source
optima were unstable (Washington Post val optimum 0.127 vs global 0.02 on n=817; shrinkage
pulled it to 0.069), confirming overfitting risk. We **ship the simpler global isotonic
calibrator** and report source-aware only as an ablation that did not beat it.

*Methods disclosure:* the test split has **no retrieved evidence**, so the 7 evidence/NLI
scalars are **imputed to TRAIN means for both the val calibration set and test**, matching the
two pipelines exactly (this is why the matched-pipeline val AUC reads **0.924**, below the
0.948 obtained when val uses its real evidence scalars). The calibrator must be fit on
probabilities generated the same way the test probabilities are, or it would not transfer.

### MMFakeBench (out-of-domain) — honest fusion
| Model | AUC (95% CI) | orig/mis/txt/vis acc |
|---|---|---|
| always-Fake floor | — (acc 0.700) | 0.00/1.00/1.00/1.00 |
| GBM [clip_prob, clip_sim, ateeq] | 0.792 [0.764, 0.823] | 0.79/0.58/0.65/0.78 |
| GBM [clip_prob, clip_sim] (shortcut-free) | 0.706 [0.673, 0.743] | 0.67/0.75/0.57/0.67 |
| AITR + Ateeq | 0.736 [0.702, 0.771] | 0.75/0.58/0.69/0.81 |

### DeBERTa NLI and its fusion lift
| Item | Number |
|---|---|
| DeBERTa v2 standalone AUC (buggy premise) | 0.4943 (chance) |
| DeBERTa v3 standalone AUC (corrected) | 0.6885 |
| NLI lift over CLIP in fusion (ΔAUC) | +0.0007 to +0.0015 (CIs overlap → none) |

### Veto
| Veto | Accuracy | Deployable |
|---|---|---|
| Oracle (ground-truth category) | 0.798 | No |
| Category-agnostic | 0.537 (prec 0.23) | Yes |

---

## 2. Claims to update (old → honest)

**Primary result.**
- ~~"86.48% on the NewsCLIPpings test set; beats SNIFFER (88.40%)."~~
- → **"An in-domain CLIP image–caption consistency detector achieving TEST AUC 0.932
  and accuracy 0.865 at a blind, val-derived isotonic-calibrated 0.5 decision, using no
  external APIs or evidence at test time."** The 86.48% was obtained by tuning the
  threshold on the test labels (data snooping) and a broken feature path (missing
  scaler); both are fixed. The honest accuracy is **0.865 [0.856, 0.873]** at the
  calibrated operating point (raw-frozen 0.5523 gave 0.831; the lift is calibration,
  not snooping). Against SNIFFER's reported 88.40% the gap is now **~1.9 pp** — and ours
  is **internal-only** vs SNIFFER's **external-retrieval/LLM** pipeline. Do **not** claim a
  head-to-head win until SNIFFER's split/protocol is verified (see `comparison_table.md`);
  reframe as "competitive, no external API," **leading with AUC** where protocols differ.

**Fusion architecture.**
- ~~"AITR transformer fusion is the key contribution and outperforms simple fusion."~~
- → **"Mixed: on MMFakeBench a gradient-boosted model on 3 scalars beats AITR
  (AUC 0.792 vs 0.736); on NewsCLIPpings AITR's embedding tokens give a small edge
  over scalar-only fusion (0.948 vs 0.930). The transformer is not a consistent win
  and adds little where CLIP scalars already separate the classes."** (Negative/mixed
  architecture finding — report honestly.)

**Text NLI.**
- ~~"Text NLI is ineffective on NewsCLIPpings."~~
- → **"The earlier non-discrimination was a scoring bug (premise = the caption's own
  article, which trivially entails). With the correct premise (the image's article),
  DeBERTa v3 is weakly discriminative (standalone AUC 0.69), but adds no measurable
  lift over CLIP in fusion (ΔAUC ≈ 0)."**

**AI-origin (Ateeq).**
- ~~"Image-origin detection contributes to fake detection."~~ / ~~"Ateeq is a non-generalizable source-pool shortcut."~~
- → **"Ateeq is a genuine AI-image detector that GENERALIZES to unseen generators, but
  carries a resolution bias that produces false positives on large, high-quality real
  photos."** A wild-image sanity test (Phase 0–3, `scripts/ateeq_wild_test.py`,
  `results/ateeq_wild_test.csv`) settled the earlier shortcut suspicion:
  - It scored never-before-seen **StyleGAN2 (recall@0.5 = 1.00)** and **Midjourney v6 (0.85)**
    images *at least as high as* its in-distribution Fakeddit training pool (0.67). A pure
    source-pool shortcut cannot do that. Wild AUC (external generators) = **0.864** ≈ control
    AUC **0.869** (positive control reproduced ~94% recall). → **NOT a pool shortcut.**
  - BUT `ai_score` correlates **r ≈ 0.6** with image resolution/file-size on real images;
    every confidently-wrong wild image was a large, high-res authentic photo (e.g. Wikimedia
    originals at 0.89–0.93). Within-real source-separation AUC peaks at 0.75 for the high-res
    Wikimedia pool → a **resolution/processing confound on real images**, not on AI-ness.
  - Caveat: small wild sample (17 real; 2 external generator families, StyleGAN faces-only),
    so "generalizes" is demonstrated for these generators, not proven universally.
- **On NewsCLIPpings the MMFakeBench fusion numbers are unchanged** (Ateeq still adds no robust
  lift there — real photos give near-zero variance, GBM 0.792 → 0.706 without it). The wild
  test corrects the *characterization* of Ateeq (it works as an AI detector), not its
  irrelevance to the out-of-context task. **In the demo, Ateeq is the image-origin signal for
  wild user uploads** (it replaced SightEngine).

**MMFakeBench veto.**
- ~~"A Wikipedia-NLI veto recovers the 'original' category (~79.8%)."~~
- → **"The veto gates on the ground-truth MMFakeBench category, which is unavailable at
  inference; it is an oracle upper bound (0.798), not deployable. The deployable,
  category-agnostic version has precision 0.23 and accuracy 0.537. It also reads the
  article-NLI score, not a Wikipedia score."**

**Generalization.**
- ~~"The system generalizes to MMFakeBench."~~
- → **"Generalization is limited: the shortcut-free MMFakeBench fusion (AUC 0.706) sits
  near the always-Fake floor (0.700). The in-domain CLIP signal does not transfer
  strongly to the out-of-domain distortion task."**

---

## 3. Limitations

The evaluation assumes a **50/50 real/fake balance**; deployment distributions are
typically imbalanced, where the calibrated 0.5 operating point and the reported accuracy
would shift (AUC is more stable but operating-point selection becomes critical, and the
isotonic calibrator is fit on a balanced val split). At the **calibrated 0.5** decision the
error profile is mildly **false-negative heavy** (FN 0.160 vs FP 0.111) — out-of-context
pairs missed — which inverts the false-positive skew of the old raw-frozen 0.5523 threshold;
either skew carries an asymmetric cost worth tuning per deployment. There is still a
**per-source failure mode** — BBC remains the weakest source (TEST acc 0.845 under global
isotonic, up from 0.796 at the frozen threshold) — indicating residual source/geographic bias.
Generalization rests on a **single external dataset** (MMFakeBench), where results are
near the majority-class floor once shortcuts are removed; broader cross-dataset testing
is needed before any general-purpose claim.

---

## 4. What survives a hostile examiner

- **Survives:** the in-domain fine-tuned CLIP image–caption consistency detector —
  TEST **AUC 0.932 / acc 0.865 at a blind, val-derived isotonic-calibrated 0.5 decision**
  (ECE 0.138 → 0.017), fresh-vs-precomputed parity r = 1.0, no test-set tuning, no external
  APIs. The accuracy lift over the old 0.831 is honest calibration, not snooping; AUC is
  unchanged. This is the defensible core.
- **Reframed, not lost:** the corrected DeBERTa v3 result (real but weak signal), and
  the honest, AUC-led fusion comparisons with confidence intervals.
- **Must be presented as negative/honest findings (or an examiner will):** AITR is not a
  consistent improvement over simple fusion; Ateeq is a source-pool shortcut; the
  MMFakeBench veto is an oracle, not deployable; out-of-domain generalization is near the
  majority floor; and the original 86.48% test headline was data-snooped and is withdrawn.
- **Net:** lead with the honest in-domain AUC, present the rest as rigorous ablations and
  limitations. A defense built on AUC + a val-derived calibrated operating point (0.865) +
  CIs + the shortcut audit is far stronger than the original inflated point estimates. The
  source-aware threshold idea is reported as a **failed ablation** (within noise of global).
