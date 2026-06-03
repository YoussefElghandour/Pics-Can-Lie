"""
evaluate.py — metrics for the Claude web-search fact-checker, plus a comparison
against the pure-NLI (DeBERTa) baseline on the SAME samples.

Reads every cached result in factcheck_llm/cache/, scores distortion detection
(positive class = "distorted"), and writes factcheck_llm/results.json.

Usage:
    python factcheck_llm/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_factcheck import CACHE_DIR, RESULTS, sample_id_from_path

# Pure-NLI baseline inputs (article ⊨ caption entailment). Low entailment ->
# predicted "distorted".  Sweep the threshold to give NLI its best possible shot.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root
import config
SAMPLE_IDS_CSV = config.MMFB_TRAIN / "val_features" / "sample_ids.csv"
DEBERTA_CSV    = config.MMFB_TRAIN / "deberta_nli_val.csv"


def load_cache() -> list[dict]:
    recs = []
    for p in sorted(CACHE_DIR.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            recs.append(json.load(f))
    return recs


def binary_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """positive class = 'distorted'."""
    tp = sum(t == "distorted" and p == "distorted" for t, p in zip(y_true, y_pred))
    fp = sum(t == "factual"   and p == "distorted" for t, p in zip(y_true, y_pred))
    fn = sum(t == "distorted" and p == "factual"   for t, p in zip(y_true, y_pred))
    tn = sum(t == "factual"   and p == "factual"   for t, p in zip(y_true, y_pred))
    n = len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / n if n else 0.0
    return {
        "n": n, "accuracy": round(accuracy, 4),
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "confusion_matrix": {
            "true_factual":   {"pred_factual": tn, "pred_distorted": fp},
            "true_distorted": {"pred_factual": fn, "pred_distorted": tp},
        },
    }


def nli_baseline(records: list[dict]) -> dict | None:
    """Best-threshold DeBERTa-NLI baseline on exactly the cached samples."""
    if not (SAMPLE_IDS_CSV.exists() and DEBERTA_CSV.exists()):
        return None
    ids = pd.read_csv(SAMPLE_IDS_CSV)[["sample_id", "image_path"]]
    deb = pd.read_csv(DEBERTA_CSV)[["sample_id", "deberta_score"]]
    merged = ids.merge(deb, on="sample_id", how="inner")
    merged["fc_id"] = merged["image_path"].map(sample_id_from_path)
    score_by_id = dict(zip(merged["fc_id"], merged["deberta_score"]))

    pairs = [(r["true_label"], score_by_id.get(r["sample_id"]))
             for r in records if r["sample_id"] in score_by_id]
    if not pairs:
        return None
    y_true = [t for t, _ in pairs]
    scores = np.array([s for _, s in pairs], dtype=float)

    # Low entailment => distorted. Sweep threshold for best accuracy.
    best = None
    for thr in np.arange(0.05, 0.96, 0.01):
        y_pred = ["distorted" if s < thr else "factual" for s in scores]
        m = binary_metrics(y_true, y_pred)
        if best is None or m["accuracy"] > best["accuracy"]:
            best = {**m, "threshold": round(float(thr), 2)}
    best["matched_samples"] = len(pairs)
    best["note"] = "DeBERTa entailment < threshold => distorted (best-accuracy threshold)"
    return best


def main() -> None:
    records = load_cache()
    if not records:
        print(f"No cached results in {CACHE_DIR}. Run run_factcheck.py first.")
        return

    n_total = len(records)
    errors  = [r for r in records if r.get("status") != "ok" or r.get("verdict") not in ("factual", "distorted")]
    valid   = [r for r in records if r not in errors]

    y_true = [r["true_label"] for r in valid]
    y_pred = [r["verdict"]    for r in valid]

    overall = binary_metrics(y_true, y_pred)

    # Accuracy over ALL cached samples, counting errors as wrong (honest view).
    correct_incl_err = sum(r.get("verdict") == r["true_label"] for r in records)
    accuracy_incl_errors = round(correct_incl_err / n_total, 4)

    # Per-category breakdown.
    per_category = {}
    for cat in ("original", "textual_veracity_distortion"):
        sub = [r for r in valid if r["fake_cls"] == cat]
        if sub:
            acc = sum(r["verdict"] == r["true_label"] for r in sub) / len(sub)
            per_category[cat] = {"n": len(sub), "accuracy": round(acc, 4)}

    baseline = nli_baseline(records)

    results = {
        "n_total_cached":        n_total,
        "n_valid":               len(valid),
        "n_errors":              len(errors),
        "model":                 records[0].get("model"),
        "distortion_detection":  overall,
        "accuracy_incl_errors":  accuracy_incl_errors,
        "per_category":          per_category,
        "nli_baseline":          baseline,
    }

    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Print headline ────────────────────────────────────────────────────────
    print(f"Cached: {n_total}   valid: {len(valid)}   errors: {len(errors)}")
    print("\n=== Distortion detection (Claude + web_search) ===")
    print(f"  accuracy   {overall['accuracy']:.4f}")
    print(f"  precision  {overall['precision']:.4f}")
    print(f"  recall     {overall['recall']:.4f}")
    print(f"  f1         {overall['f1']:.4f}")
    cm = overall["confusion_matrix"]
    print(f"  confusion  true_factual:   pred_factual={cm['true_factual']['pred_factual']:>4}  pred_distorted={cm['true_factual']['pred_distorted']:>4}")
    print(f"             true_distorted: pred_factual={cm['true_distorted']['pred_factual']:>4}  pred_distorted={cm['true_distorted']['pred_distorted']:>4}")
    print("\n  per category:")
    for cat, d in per_category.items():
        print(f"    {cat:<28} acc={d['accuracy']:.4f}  (n={d['n']})")
    if baseline:
        print("\n=== Pure-NLI baseline (DeBERTa, best threshold) ===")
        print(f"  accuracy {baseline['accuracy']:.4f}  f1 {baseline['f1']:.4f}  "
              f"@thr={baseline['threshold']}  (matched {baseline['matched_samples']})")
        print(f"  --> LLM vs NLI accuracy delta: {overall['accuracy'] - baseline['accuracy']:+.4f}")
    else:
        print("\n(NLI baseline unavailable — score CSVs not found / no id overlap.)")
    print(f"\nSaved -> {RESULTS}")


if __name__ == "__main__":
    main()
