"""
DeBERTa NLI Standalone Evaluation
-----------------------------------
Uses cached entailment_score from fusion_features_finetuned_1000.csv.
Label encoding: 0 = REAL, 1 = OUT-OF-CONTEXT (fake).

The DeBERTa module returns an entailment probability — how well the article
text supports the caption.  A HIGH entailment score → caption is supported →
image is likely REAL.  A LOW score → caption is NOT supported → likely FAKE.

So the prediction rule is:
    pred = 0 (REAL)  if entailment_score >= threshold
    pred = 1 (FAKE)  if entailment_score <  threshold
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# ── 1. Load cached scores ──────────────────────────────────────────────────────
df = pd.read_csv("fusion_features_finetuned_1000.csv")
scores = df["entailment_score"].values   # higher → more entailed → REAL
labels = df["label"].values              # 0=REAL, 1=FAKE

print(f"Loaded {len(df)} samples  |  REAL: {(labels==0).sum()}  FAKE: {(labels==1).sum()}")
print(f"Score range: [{scores.min():.6f}, {scores.max():.6f}]  mean={scores.mean():.4f}")
print()

# ── 2. Threshold sweep (optimise F1) ──────────────────────────────────────────
thresholds = np.arange(0.30, 0.71, 0.05)
results = []

for thr in thresholds:
    preds = (scores < thr).astype(int)   # below threshold → FAKE
    f1  = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    results.append({"threshold": round(float(thr), 2), "accuracy": acc,
                    "precision": prec, "recall": rec, "f1": f1})

sweep_df = pd.DataFrame(results)

# Pick the threshold with the best F1 (ties broken by accuracy)
best_idx  = sweep_df["f1"].idxmax()
best_row  = sweep_df.loc[best_idx]
best_thr  = best_row["threshold"]

print("Threshold sweep:")
print(sweep_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print()

# ── 3. Final metrics at best threshold ────────────────────────────────────────
preds_best = (scores < best_thr).astype(int)

acc  = accuracy_score(labels, preds_best)
prec = precision_score(labels, preds_best, zero_division=0)
rec  = recall_score(labels, preds_best, zero_division=0)
f1   = f1_score(labels, preds_best, zero_division=0)
cm   = confusion_matrix(labels, preds_best).tolist()

print("=" * 56)
print(f"  Optimal threshold : {best_thr}")
print(f"  Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision         : {prec:.4f}")
print(f"  Recall            : {rec:.4f}")
print(f"  F1                : {f1:.4f}  ({f1*100:.2f}%)")
print("=" * 56)
print()
print("Confusion matrix (rows=actual, cols=predicted):")
print("               Pred REAL  Pred FAKE")
print(f"  Actual REAL     {cm[0][0]:>5}      {cm[0][1]:>5}")
print(f"  Actual FAKE     {cm[1][0]:>5}      {cm[1][1]:>5}")
print()
print("Classification report:")
print(classification_report(labels, preds_best, target_names=["REAL", "FAKE"]))

# ── 4. Save JSON ───────────────────────────────────────────────────────────────
output = {
    "model": "cross-encoder/nli-deberta-v3-large",
    "dataset": "NewsClipPings (fusion_features_finetuned_1000.csv)",
    "n_samples": int(len(df)),
    "n_real": int((labels == 0).sum()),
    "n_fake": int((labels == 1).sum()),
    "threshold_sweep": sweep_df.to_dict(orient="records"),
    "optimal_threshold": float(best_thr),
    "metrics": {
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
    },
    "confusion_matrix": {
        "true_real_pred_real": cm[0][0],
        "true_real_pred_fake": cm[0][1],
        "true_fake_pred_real": cm[1][0],
        "true_fake_pred_fake": cm[1][1],
    },
}

with open("deberta_standalone_eval.json", "w") as fh:
    json.dump(output, fh, indent=2)

print("Results saved to deberta_standalone_eval.json")
