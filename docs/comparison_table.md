# Pics Can Lie — Corrected Results & Comparison Table

Sources (all under `results/`): `honest_numbers_summary.json`,
`mmfakebench_honest_comparison.json`, `veto_honest_report.json`,
`deberta_v3_retest.json`, `fusion_nli_lift.json`.

**Protocol throughout:** AUC is the primary metric (threshold-free). Accuracy is
reported only at a threshold **frozen on a held-out split** (train or train-calib),
never tuned on the set it is scored on. 95% CIs are 1000-resample bootstraps.

---

## 1. My honest numbers

### NewsCLIPpings (in-domain) — fine-tuned CLIP v2 → AITR fusion
| Split | AUC (95% CI) | Accuracy (95% CI) | Threshold | Notes |
|---|---|---|---|---|
| VAL  | **0.9477** [0.9351, 0.9605] | 0.880 [0.861, 0.900] | 0.5523 (train-frozen) | internal val split |
| TEST | **0.9317** [0.9258, 0.9371] | **0.8308** [0.8221, 0.8392] | 0.5523 (frozen, **blind**) | 7264 samples, fresh features |
| ~~TEST (invalid)~~ | — | ~~0.8648~~ | 0.010 (swept on TEST) | data-snooped — discarded |

Per-source TEST acc @0.5523: BBC 0.796 · Guardian 0.832 · USA Today 0.842 · Washington Post 0.843.
Fresh-vs-precomputed parity: Pearson r = 1.0000 (test pipeline ≡ val pipeline).

### MMFakeBench (out-of-domain generalization) — honest fusion comparison
| Model | AUC (95% CI) | Acc @train-frozen | orig / mis / txt / vis acc |
|---|---|---|---|
| always-predict-Fake (floor) | — | 0.700 | 0.00 / 1.00 / 1.00 / 1.00 |
| AITR + Ateeq (Variant B) | 0.736 [0.702, 0.771] | 0.687 | 0.75 / 0.58 / 0.69 / 0.81 |
| **GBM [clip_prob, clip_sim, ateeq]** | **0.792** [0.764, 0.823] | 0.685 | 0.79 / 0.58 / 0.65 / 0.78 |
| GBM [clip_prob, clip_sim] (no Ateeq) | 0.706 [0.673, 0.743] | 0.666 | 0.67 / 0.75 / 0.57 / 0.67 |
| LogReg [clip_prob, clip_sim, ateeq] | 0.747 [0.714, 0.781] | 0.691 | 0.69 / 0.66 / 0.67 / 0.85 |

Best deployable MMFakeBench model (GBM, AUC 0.792) is only ~9pp over the always-Fake
floor; shortcut-free (no-Ateeq) it drops to AUC 0.706 — near the floor. The Ateeq
lift is a source-pool shortcut (0.258 ai-ness gap on real-only images), not robust signal.

### DeBERTa NLI — scoring-bug correction
| Version | REAL mean | FAKE mean | Standalone AUC |
|---|---|---|---|
| v2 (wrong premise = caption's own article) | 0.4150 | 0.4216 | 0.4943 (chance) |
| **v3 (correct premise = image's article)** | 0.3181 | 0.0545 | **0.6885** |

The "text NLI is ineffective" conclusion was a **scoring bug**, not a task property.

### NLI fusion lift (does v3 NLI help over CLIP?)
| Features | LogReg AUC | GBM AUC |
|---|---|---|
| CLIP-only [prob, sim] | 0.9273 | 0.9295 |
| CLIP + v3 NLI [prob, sim, deberta_v3] | 0.9288 | 0.9302 |
| **Δ from adding v3 NLI** | **+0.0015** | **+0.0007** |

NLI adds **no measurable lift** over CLIP (ΔAUC ≈ 0, CIs fully overlap). CLIP already
captures the consistency signal. Note: best simple scalar fusion (0.930) is ~1.75pp
**below** AITR (0.948) on NewsCLIPpings — the embeddings help here (unlike MMFakeBench,
where GBM beats AITR).

### MMFakeBench veto status
| | Accuracy | Deployable? |
|---|---|---|
| Oracle (uses ground-truth `fake_cls`) | 0.798 | **NO** — ground-truth metadata, not available at inference |
| Category-agnostic (deployable) | 0.537 | yes — precision 0.23 |

---

## 2. Literature comparison — VERIFY protocol before claiming anything

> Do **not** assert "beaten" or "below" until each row's protocol is confirmed in the
> original paper. My honest TEST is **0.831 acc / 0.932 AUC at a blind frozen threshold,
> no external APIs/evidence at test time.** Where protocols differ, lead with AUC.

| Method | Reported acc | Split / protocol to verify before comparing |
|---|---|---|
| **Ours (CLIP→AITR)** | **0.831 (AUC 0.932)** | NewsCLIPpings **test**, balanced, threshold **frozen on train** (blind), no external API at test |
| SNIFFER | 88.40% | Which split (test vs val)? Balanced vs full? Threshold tuning method? Uses an external LLM/API or evidence retrieval? |
| COSMOS | 85.00% | Same NewsCLIPpings split? Self-supervised / out-of-context protocol — does it use the same balanced test set and labels? |
| MUSE-MLP | 90.00% | Which split/balance? Threshold selection (val-frozen vs swept)? Same train/eval partition? |
| MUSE-AITR | 93.30% | Same as MUSE-MLP — confirm split, balance, threshold protocol, and any evidence/external inputs |
| RED-DOT | 90.30% | Uses retrieved external evidence? Which split/balance? Threshold method? |
| MIRAGE | 75.10% | Split/balance and metric definition — confirm it is the comparable NewsCLIPpings test setup |

**Specifically at risk:** the "beats SNIFFER" claim — my honest TEST (0.831) is below
SNIFFER's reported 88.40%. Before any comparison: confirm SNIFFER's split (test vs val),
balanced-vs-full, threshold-tuning method, and whether it uses an external LLM/evidence
pipeline. If SNIFFER uses an external LLM and a different protocol, reframe to
"**competitive in-domain detector with no external API**, leading with AUC where
protocols differ" rather than a head-to-head accuracy win.

*(No external paper's protocol is asserted here — each row lists only what must be checked.)*
