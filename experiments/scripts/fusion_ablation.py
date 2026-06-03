"""
Fusion Ablation Study — cached scores only, no API calls or model loading.

Data source: fusion_features_300_3feat_finetuned.csv
  - itm_score            : fine-tuned BLIP ITM (epoch 9, F1=75.1%)
  - entailment_score     : DeBERTa NLI entailment probability
  - sightengine_score    : SightEngine AI-generated image probability
  - label                : 0=REAL, 1=FAKE (150/150 balanced)

All 5 combinations are evaluated on the same 300-sample set so results are
directly comparable.  The 82.67% baseline is the pretrained-BLIP 3-module
fusion on this same 300-sample set.
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("fusion_features_300_3feat_finetuned.csv")

print("Columns:", df.columns.tolist())
print(f"Shape  : {df.shape}")
print(f"Labels : REAL={( df['label']==0).sum()}  FAKE={(df['label']==1).sum()}")
print()

y  = df["label"].values
X_blip   = df[["itm_score"]].values
X_deb    = df[["entailment_score"]].values
X_se     = df[["sightengine_score"]].values
X_blip_se = df[["itm_score", "sightengine_score"]].values
X_blip_deb = df[["itm_score", "entailment_score"]].values
X_all    = df[["itm_score", "entailment_score", "sightengine_score"]].values

# ── CV setup ──────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "f1", "precision", "recall"]

COMBINATIONS = [
    ("BLIP only",                X_blip),
    ("DeBERTa only",             X_deb),
    ("BLIP + SightEngine",       X_blip_se),
    ("BLIP + DeBERTa",          X_blip_deb),
    ("BLIP + DeBERTa + SightEngine", X_all),
]

# ── Run CV for each combination ───────────────────────────────────────────────
records = []
print(f"{'Combination':<35} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
print("-" * 75)

for name, X in COMBINATIONS:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
    ])
    cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring)

    acc  = cv_res["test_accuracy"].mean()
    f1   = cv_res["test_f1"].mean()
    prec = cv_res["test_precision"].mean()
    rec  = cv_res["test_recall"].mean()

    records.append({
        "combination":  name,
        "accuracy":     round(acc,  4),
        "f1":           round(f1,   4),
        "precision":    round(prec, 4),
        "recall":       round(rec,  4),
        "accuracy_pct": round(acc  * 100, 2),
        "f1_pct":       round(f1   * 100, 2),
    })
    print(f"{name:<35} {acc*100:>9.2f}% {f1*100:>9.2f}% {prec*100:>9.2f}% {rec*100:>9.2f}%")

# ── Ranked summary (by F1) ─────────────────────────────────────────────────────
results_df = pd.DataFrame(records).sort_values("f1", ascending=False).reset_index(drop=True)
results_df.insert(0, "rank", results_df.index + 1)

print()
print("=" * 75)
print("RANKED SUMMARY  (sorted by F1, 5-fold CV, n=300 balanced samples)")
print("=" * 75)
print(f"{'Rank':<5} {'Combination':<35} {'Accuracy':>10} {'F1':>10}")
print("-" * 62)
BASELINE = 82.67
for _, row in results_df.iterrows():
    flag = "  <- baseline" if row["combination"] == "BLIP + DeBERTa + SightEngine" else ""
    beat = "  ** beats 82.67%**" if row["f1_pct"] > BASELINE and flag == "" else ""
    print(f"{int(row['rank']):<5} {row['combination']:<35} {row['accuracy_pct']:>9.2f}% {row['f1_pct']:>9.2f}%{flag}{beat}")
print("-" * 62)
print(f"      {'Pretrained 3-module baseline':<35} {'~82.67%':>10} {'~82.67%':>10}  (reported in README)")
print()

# ── Save JSON ──────────────────────────────────────────────────────────────────
output = {
    "description": "Fusion ablation study — 5-fold LR CV on cached scores, n=300 balanced",
    "data_source":  "fusion_features_300_3feat_finetuned.csv",
    "n_samples":    int(len(df)),
    "n_real":       int((y == 0).sum()),
    "n_fake":       int((y == 1).sum()),
    "cv_folds":     5,
    "baseline_accuracy_pct": BASELINE,
    "baseline_f1_pct":       BASELINE,
    "results": records,
    "ranked_by_f1": results_df.drop(columns=["accuracy_pct","f1_pct"]).to_dict(orient="records"),
}

with open("ablation_results.json", "w") as fh:
    json.dump(output, fh, indent=2)

print("Full results saved to ablation_results.json")
