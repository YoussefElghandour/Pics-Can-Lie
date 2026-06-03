"""
Kaggle export — Option B (full)
Copies all 71k resolvable training images plus the two JSON files
into a single upload-ready folder.

Output structure:
    D:/Pics Can Lie/kaggle_dataset_full/
    ├── merged_balanced/train.json
    ├── metadata/train.json
    └── images/<source>/<subpath>/image.jpg   (mirrors original relative path)

Usage:
    python kaggle_export_full.py
"""
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402


import json
import os
import shutil

# ── Config ────────────────────────────────────────────────────────────────────
ANNOTATIONS  = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'merged_balanced', 'train.json')
METADATA     = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'metadata', 'train.json')
IMAGES_ROOT  = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'origin')
OUTPUT_DIR   = _os.path.join(str(_cfg.ROOT), 'kaggle_dataset_full')
LOG_EVERY    = 2000

# ── Load annotations + metadata ───────────────────────────────────────────────
print("Loading annotations …")
with open(ANNOTATIONS, encoding="utf-8") as f:
    all_annotations = json.load(f)["annotations"]

print("Loading metadata …")
with open(METADATA, encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Total annotations: {len(all_annotations):,}")

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
copied = skipped = missing = 0

for i, ann in enumerate(all_annotations, start=1):
    image_id = str(ann["image_id"])

    if image_id not in metadata:
        missing += 1
        continue

    rel       = metadata[image_id]["image_path"].replace("visual_news/", "", 1)
    full_path = os.path.join(IMAGES_ROOT, rel)

    if not os.path.exists(full_path):
        missing += 1
        continue

    dest = os.path.join(images_dir, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    try:
        shutil.copy2(full_path, dest)
        copied += 1
    except Exception as e:
        print(f"  SKIP {rel}: {e}")
        skipped += 1

    if i % LOG_EVERY == 0 or i == len(all_annotations):
        print(f"  {i:,}/{len(all_annotations):,}  copied={copied:,}  missing={missing:,}  skipped={skipped:,}")

# ── Summary ───────────────────────────────────────────────────────────────────
total_mb = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, files in os.walk(OUTPUT_DIR)
    for f in files
) / 1024 / 1024

print(f"\nDone.")
print(f"  Copied : {copied:,} images")
print(f"  Missing: {missing:,}")
print(f"  Skipped: {skipped:,}")
print(f"  Output : {OUTPUT_DIR}")
print(f"  Size   : {total_mb:.0f} MB ({total_mb / 1024:.2f} GB)")
