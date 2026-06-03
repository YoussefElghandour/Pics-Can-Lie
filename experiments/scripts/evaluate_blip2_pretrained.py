"""
BLIP-2 zero-shot evaluation — Pics Can Lie
Model  : Salesforce/blip2-opt-2.7b  (pretrained, no fine-tuning)
Task   : Classify image-caption pairs as real (0) or out-of-context (1)
Dataset: D:/Pics Can Lie/merged_balanced/train.json

Usage:
    python evaluate_blip2_pretrained.py
"""
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402


import json
import os
import random

import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "Salesforce/blip2-opt-2.7b"
ANNOTATIONS     = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'merged_balanced', 'train.json')
METADATA        = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'metadata', 'train.json')
IMAGES_ROOT     = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'origin')
OUTPUT_PATH     = _os.path.join(str(_cfg.ROOT), 'results', 'blip2_pretrained_eval.json')
MAX_SAMPLES     = 500
SEED            = 42
MAX_NEW_TOKENS  = 20

# ── Load dataset ──────────────────────────────────────────────────────────────

def load_samples(annotations_path: str, metadata_path: str, images_root: str,
                 max_samples: int, seed: int) -> list[dict]:
    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)  # keyed by image_id (string)

    joined = []
    for ann in annotations:
        image_id = str(ann["image_id"])
        if image_id not in metadata:
            continue
        meta    = metadata[image_id]
        caption = meta.get("caption") or meta.get("title") or ""
        rel     = meta["image_path"].replace("visual_news/", "", 1)
        joined.append({
            "image_path": os.path.join(images_root, rel),
            "caption":    caption,
            "label":      int(ann["falsified"]),
        })

    random.seed(seed)
    return random.sample(joined, min(max_samples, len(joined)))


# ── Classification rule ───────────────────────────────────────────────────────

def classify_response(response: str) -> int:
    """Return 1 (out-of-context) if response signals mismatch, else 0 (real)."""
    lower = response.lower().strip()
    if any(kw in lower for kw in ("out-of-context", "no", "false")):
        return 1
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}  |  dtype: {dtype}")

    print("Loading processor …")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Loading model …")
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    print("Model ready.\n")

    print("Loading dataset …")
    samples = load_samples(ANNOTATIONS, METADATA, IMAGES_ROOT, MAX_SAMPLES, SEED)
    print(f"Evaluating {len(samples)} samples\n")

    results      = []
    y_true       = []
    y_pred       = []
    skipped      = 0

    for i, item in enumerate(samples, start=1):
        # Load image
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            print(f"[{i}/{len(samples)}] SKIP — cannot open image: {e}")
            skipped += 1
            continue

        prompt = (
            f"Does this image match the caption: '{item['caption'][:100]}'? "
            "Answer:"
        )

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(device, dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        response  = processor.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()
        predicted = classify_response(response)

        y_true.append(item["label"])
        y_pred.append(predicted)

        results.append({
            "image_path": item["image_path"],
            "caption":    item["caption"],
            "label":      item["label"],
            "response":   response,
            "predicted":  predicted,
            "correct":    predicted == item["label"],
        })

        if i % 50 == 0 or i == len(samples):
            running_acc = accuracy_score(y_true, y_pred)
            print(f"[{i}/{len(samples)}]  running accuracy: {running_acc:.3f}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred).tolist()

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Skipped  : {skipped}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred,
                                target_names=["real", "out-of-context"],
                                zero_division=0))

    # ── Save output ───────────────────────────────────────────────────────────
    output = {
        "config": {
            "model":       MODEL_ID,
            "annotations": ANNOTATIONS,
            "max_samples": MAX_SAMPLES,
            "seed":        SEED,
            "evaluated":   len(y_true),
            "skipped":     skipped,
        },
        "metrics": {
            "accuracy":         acc,
            "precision":        prec,
            "recall":           rec,
            "f1":               f1,
            "confusion_matrix": cm,
        },
        "predictions": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
