"""
Batch evaluation script — Pics Can Lie
Loads 100 balanced samples (50 real + 50 fake) from the NewsClipPings test set,
runs each through the full pipeline, and prints accuracy, precision, recall, F1,
and a full classification report.

Usage:
    python evaluate.py
"""

import json
import os

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

load_dotenv()

# ── Paths ──
DATASET_ROOT     = os.path.join(os.path.dirname(__file__), "dataset")
ANNOTATIONS_PATH = os.path.join(DATASET_ROOT, "data", "NewsClipPings", "merged_balanced", "test.json")
METADATA_PATH    = os.path.join(DATASET_ROOT, "data", "NewsClipPings", "metadata", "test.json")
IMAGE_BASE       = os.path.join(DATASET_ROOT, "origin")
ARTICLE_BASE     = os.path.join(DATASET_ROOT, "origin")

SAMPLES_PER_CLASS = 50  # 50 real + 50 fake = 100 total
ITM_THRESHOLD = 0.5  # predict Out-of-Context if itm_score < threshold


# ── Dataset loading ───────────────────────────────────────────────────────────

def resolve_image_path(meta_image_path: str) -> str:
    rel = meta_image_path.replace("visual_news/", "", 1)
    return os.path.join(IMAGE_BASE, rel)


def resolve_article_path(meta_article_path: str) -> str:
    rel = meta_article_path.replace("visual_news/", "", 1)
    return os.path.join(ARTICLE_BASE, rel)


def load_samples() -> list[dict]:
    """Load SAMPLES_PER_CLASS real and SAMPLES_PER_CLASS fake samples from the test set."""
    print("Loading test annotations and metadata...")
    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    real_samples, fake_samples = [], []

    for ann in annotations:
        if len(real_samples) >= SAMPLES_PER_CLASS and len(fake_samples) >= SAMPLES_PER_CLASS:
            break

        art_id = str(ann["id"])
        img_id = str(ann["image_id"])

        if art_id not in metadata or img_id not in metadata:
            continue

        img_path = resolve_image_path(metadata[img_id]["image_path"])
        art_path = resolve_article_path(metadata[img_id]["article_path"])

        if not os.path.isfile(img_path) or not os.path.isfile(art_path):
            continue

        try:
            with open(art_path, "r", encoding="utf-8", errors="ignore") as f:
                article_text = f.read().strip()
        except Exception:
            continue

        sample = {
            "article_id": art_id,
            "image_id":   img_id,
            "image_path": img_path,
            "caption":    metadata[img_id]["caption"],
            "article_text": article_text,
            "falsified":  ann["falsified"],
        }

        if ann["falsified"] and len(fake_samples) < SAMPLES_PER_CLASS:
            fake_samples.append(sample)
        elif not ann["falsified"] and len(real_samples) < SAMPLES_PER_CLASS:
            real_samples.append(sample)

    samples = real_samples + fake_samples
    print(f"Loaded {len(real_samples)} real + {len(fake_samples)} fake = {len(samples)} total samples")
    return samples


# ── Per-sample prediction ─────────────────────────────────────────────────────

def predict(
    sample: dict,
    get_blip_itm_score,
    get_nli_entailment_score,
    get_sightengine_score,
) -> tuple[int, float, float, float, bool]:
    """Run the full pipeline on one sample.
    Returns (pred, itm_score, nli_score, se_score, se_was_defaulted).
    Prediction rule: Out-of-Context (1) if itm_score < ITM_THRESHOLD, else Consistent (0).
    """
    image        = Image.open(sample["image_path"]).convert("RGB")
    caption      = sample["caption"]
    article_text = sample["article_text"]

    itm_score = get_blip_itm_score(image, caption)
    nli_score = get_nli_entailment_score(article_text, caption) if article_text else 0.0
    raw_se    = get_sightengine_score(image)
    se_defaulted = raw_se < 0
    se_score  = 0.0 if se_defaulted else raw_se

    pred = 1 if itm_score < ITM_THRESHOLD else 0
    return pred, itm_score, nli_score, se_score, se_defaulted


# ── Main ──────────────────────────────────────────────────────────────────────

def sightengine_debug(sample: dict) -> None:
    """Call SightEngine on one sample and print the full raw API response."""
    import tempfile
    import requests
    from app import SIGHTENGINE_URL, SIGHTENGINE_USER, SIGHTENGINE_SECRET

    print("\n--- SightEngine Debug ---")
    print(f"Image: {sample['image_path']}")

    image = Image.open(sample["image_path"]).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp, format="JPEG")
        tmp_path = tmp.name

    try:
        params = {
            "models": "genai",
            "api_user": SIGHTENGINE_USER,
            "api_secret": SIGHTENGINE_SECRET,
        }
        with open(tmp_path, "rb") as f:
            resp = requests.post(SIGHTENGINE_URL, files={"media": ("image.jpg", f)},
                                 data=params, timeout=30)
        print(f"HTTP status : {resp.status_code}")
        print(f"Raw response: {resp.text}")
        print("\nResponse headers:")
        for k, v in resp.headers.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Request failed: {e}")
    finally:
        os.unlink(tmp_path)
    print("--- End SightEngine Debug ---\n")


def main():
    print("Loading models (this may take a moment)...")
    from app import get_blip_itm_score, get_nli_entailment_score, get_sightengine_score

    samples = load_samples()
    sightengine_debug(samples[0])
    if not samples:
        print("No samples found. Check dataset paths.")
        return

    y_true, y_pred = [], []
    itm_scores, nli_scores, se_scores = [], [], []
    se_defaulted_count = 0
    errors = 0

    print("\n--- First 5 Sample Predictions ---")
    for sample in tqdm(samples, desc="Evaluating"):
        true_label = 1 if sample["falsified"] else 0
        y_true.append(true_label)
        try:
            pred, itm, nli, se, se_def = predict(
                sample,
                get_blip_itm_score, get_nli_entailment_score, get_sightengine_score,
            )
            y_pred.append(pred)
            itm_scores.append(itm)
            nli_scores.append(nli)
            se_scores.append(se)
            if se_def:
                se_defaulted_count += 1

            if len(y_pred) <= 5:
                true_str = "real" if true_label == 0 else "fake"
                pred_str = "real" if pred == 0 else "fake"
                correct  = "✓" if pred == true_label else "✗"
                caption  = sample["caption"][:120] + "..." if len(sample["caption"]) > 120 else sample["caption"]
                print(f"\n[{len(y_pred)}] {correct} true={true_str}  pred={pred_str}  ITM={itm:.4f}  NLI={nli:.4f}")
                print(f"    Caption: {caption}")
        except Exception as e:
            print(f"\nError on sample {sample['article_id']}: {e}")
            y_pred.append(0)
            errors += 1

    if errors:
        print(f"\nWarning: {errors} samples failed and were defaulted to 'real'.")

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS  (rule: ITM < 0.5 → fake)")
    print("=" * 50)
    print(f"Samples evaluated : {len(y_true)}  ({y_true.count(0)} real, {y_true.count(1)} fake)")
    print(f"Accuracy          : {acc:.4f}  ({acc:.1%})")
    print(f"Precision         : {prec:.4f}")
    print(f"Recall            : {rec:.4f}")
    print(f"F1 Score          : {f1:.4f}")

    itm_scores = np.array(itm_scores)
    nli_scores = np.array(nli_scores)
    se_scores  = np.array(se_scores)
    y_arr      = np.array(y_true[:len(itm_scores)])
    real_mask  = y_arr == 0
    fake_mask  = y_arr == 1

    n = len(itm_scores)
    print(f"\n--- Score Diagnostics ({n} samples) ---")
    print(f"ITM score   — mean: {itm_scores.mean():.4f}  min: {itm_scores.min():.4f}  max: {itm_scores.max():.4f}")
    print(f"NLI score   — mean: {nli_scores.mean():.4f}  min: {nli_scores.min():.4f}  max: {nli_scores.max():.4f}")
    print(f"SE score    — mean: {se_scores.mean():.4f}  min: {se_scores.min():.4f}  max: {se_scores.max():.4f}")
    print(f"SE defaulted to 0.0 (API returned -1): {se_defaulted_count}/{n} samples ({se_defaulted_count/n:.1%})")

    print(f"\n--- Score Breakdown by True Label ---")
    print(f"{'':25s}  {'Real (n=' + str(real_mask.sum()) + ')':>14}  {'Fake (n=' + str(fake_mask.sum()) + ')':>14}")
    print(f"{'ITM score (mean)':25s}  {itm_scores[real_mask].mean():>14.4f}  {itm_scores[fake_mask].mean():>14.4f}")
    print(f"{'NLI score (mean)':25s}  {nli_scores[real_mask].mean():>14.4f}  {nli_scores[fake_mask].mean():>14.4f}")
    if se_defaulted_count > n * 0.5:
        print("  ⚠  WARNING: SightEngine is failing for the majority of samples.")
        print("     The fusion model was trained with real SE scores — predictions may be distorted.")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Consistent (real)", "Out-of-Context (fake)"]))


if __name__ == "__main__":
    main()
