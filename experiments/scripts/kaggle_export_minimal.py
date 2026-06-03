"""
Kaggle export — Option A (minimal)
Copies only the ~5500 images actually used during training and evaluation,
plus the two JSON files, into a single upload-ready folder.

Output structure:
    D:/Pics Can Lie/kaggle_dataset_minimal/
    ├── merged_balanced/train.json
    ├── metadata/train.json
    └── images/<source>/<subpath>/image.jpg   (mirrors original relative path)

Usage:
    python kaggle_export_minimal.py
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
import shutil

# ── Config ────────────────────────────────────────────────────────────────────
ANNOTATIONS  = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'merged_balanced', 'train.json')
METADATA     = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'metadata', 'train.json')
IMAGES_ROOT  = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'origin')
OUTPUT_DIR   = _os.path.join(str(_cfg.ROOT), 'kaggle_dataset_minimal')
TRAIN_LIMIT  = 5000
EVAL_LIMIT   = 500
SEED         = 42

# ── Load annotations + metadata ───────────────────────────────────────────────
print("Loading annotations …")
with open(ANNOTATIONS, encoding="utf-8") as f:
    all_annotations = json.load(f)["annotations"]

print("Loading metadata …")
with open(METADATA, encoding="utf-8") as f:
    metadata = json.load(f)

# ── Resolve image paths for all annotations ───────────────────────────────────
resolved = []
for ann in all_annotations:
    image_id = str(ann["image_id"])
    if image_id not in metadata:
        continue
    rel       = metadata[image_id]["image_path"].replace("visual_news/", "", 1)
    full_path = os.path.join(IMAGES_ROOT, rel)
    if os.path.exists(full_path):
        resolved.append((ann, full_path, rel))

print(f"Resolvable annotations: {len(resolved):,}")

# ── Select eval split first (seed-fixed), then train pool ────────────────────
random.seed(SEED)
eval_indices  = set(random.sample(range(len(resolved)), min(EVAL_LIMIT, len(resolved))))
eval_items    = [resolved[i] for i in eval_indices]
train_items   = [item for i, item in enumerate(resolved) if i not in eval_indices]
train_items   = train_items[:TRAIN_LIMIT]

selected = eval_items + train_items
print(f"Selected: {len(eval_items)} eval + {len(train_items)} train = {len(selected)} total images")

# ── Copy JSON files ───────────────────────────────────────────────────────────
for src, rel_dest in [
    (ANNOTATIONS, os.path.join("merged_balanced", "train.json")),
    (METADATA,    os.path.join("metadata",        "train.json")),
]:
    dest = os.path.join(OUTPUT_DIR, rel_dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Copying {os.path.basename(src)} …")
    shutil.copy2(src, dest)

# ── Copy images ───────────────────────────────────────────────────────────────
images_dir = os.path.join(OUTPUT_DIR, "images")
copied = skipped = 0

for i, (ann, full_path, rel) in enumerate(selected, start=1):
    dest = os.path.join(images_dir, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        shutil.copy2(full_path, dest)
        copied += 1
    except Exception as e:
        print(f"  SKIP {rel}: {e}")
        skipped += 1

    if i % 500 == 0 or i == len(selected):
        print(f"  {i}/{len(selected)} images copied …")

# ── Summary ───────────────────────────────────────────────────────────────────
total_mb = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, files in os.walk(OUTPUT_DIR)
    for f in files
) / 1024 / 1024

print(f"\nDone.")
print(f"  Copied : {copied:,} images")
print(f"  Skipped: {skipped:,}")
print(f"  Output : {OUTPUT_DIR}")
print(f"  Size   : {total_mb:.0f} MB ({total_mb / 1024:.2f} GB)")
