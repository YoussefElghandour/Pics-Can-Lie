# Pics Can Lie — Corrected Results & Claims (thesis scaffold)

Single source of truth for the Results and Discussion chapters. Every number is
from the honest, audited pipeline (frozen thresholds, AUC-primary, bootstrap CIs).
Derived from `results/*.json`; full comparison in `docs/comparison_table.md`.

---

## 1. Honest results table

### NewsCLIPpings (in-domain) — CLIP v2 → AITR
| Split | AUC (95% CI) | Accuracy (95% CI) | Threshold |
|---|---|---|---|
| VAL  | 0.9477 [0.9351, 0.9605] | 0.880 [0.861, 0.900] | 0.5523 (train-frozen) |
| TEST | **0.9317 [0.9258, 0.9371]** | **0.8308 [0.8221, 0.8392]** | 0.5523 (blind) |

Invalid prior headline (threshold swept on test, 0.8648) is retired.

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
  and accuracy 0.831 at a blind, train-frozen threshold, using no external APIs or
  evidence at test time."** The 86.48% was obtained by tuning the threshold on the
  test labels (data snooping) and a broken feature path (missing scaler); both are
  fixed. Do **not** claim a head-to-head win over SNIFFER until its split/protocol is
  verified (see `comparison_table.md`); reframe as "competitive, no external API,"
  leading with AUC where protocols differ.

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
- ~~"Image-origin detection contributes to fake detection."~~
- → **"Ateeq keys on the source image pool, not manipulation: a 0.258 ai-ness gap
  between two categories that both use authentic photos. Its apparent MMFakeBench lift
  is a non-generalizable shortcut; removing it drops AUC 0.792 → 0.706 (near floor)."**

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
typically imbalanced, where a fixed threshold and the reported accuracy would shift
(AUC is more stable but operating-point selection becomes critical). The error profile
is **false-positive heavy** at the frozen threshold (real news flagged as out-of-context),
which carries a different cost than misses. There is a **per-source failure mode** — BBC
is the weakest source (TEST acc 0.796) — indicating residual source/geographic bias.
Generalization rests on a **single external dataset** (MMFakeBench), where results are
near the majority-class floor once shortcuts are removed; broader cross-dataset testing
is needed before any general-purpose claim.

---

## 4. What survives a hostile examiner

- **Survives:** the in-domain fine-tuned CLIP image–caption consistency detector —
  TEST **AUC 0.932 / acc 0.831 at a blind frozen threshold**, fresh-vs-precomputed
  parity r = 1.0, no test-set tuning, no external APIs. This is the defensible core.
- **Reframed, not lost:** the corrected DeBERTa v3 result (real but weak signal), and
  the honest, AUC-led fusion comparisons with confidence intervals.
- **Must be presented as negative/honest findings (or an examiner will):** AITR is not a
  consistent improvement over simple fusion; Ateeq is a source-pool shortcut; the
  MMFakeBench veto is an oracle, not deployable; out-of-domain generalization is near the
  majority floor; and the original 86.48% test headline was data-snooped and is withdrawn.
- **Net:** lead with the honest in-domain AUC, present the rest as rigorous ablations and
  limitations. A defense built on AUC at a frozen threshold + CIs + the shortcut audit is
  far stronger than the original inflated point estimates.
