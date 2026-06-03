import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402
import json
import os

VAL_LABELS  = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'merged_balanced', 'val.json')
VAL_META    = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'metadata', 'val.json')
IMAGES_ROOT = _os.path.join(str(_cfg.ROOT), 'kaggle_dataset_full', 'images')

with open(VAL_LABELS) as f:
    annotations = json.load(f)["annotations"]
with open(VAL_META) as f:
    metadata = json.load(f)

total    = 0
found    = 0
missing  = 0
missing_paths = []

for ann in annotations:
    key = str(ann["image_id"])
    if key not in metadata:
        continue
    total += 1
    raw      = metadata[key]["image_path"]
    stripped = raw.removeprefix("visual_news/")
    full     = os.path.join(IMAGES_ROOT, stripped)
    if os.path.exists(full):
        found += 1
    else:
        missing += 1
        if len(missing_paths) < 5:
            missing_paths.append(full)

coverage = 100 * found / total if total else 0

print(f"Total val samples : {total}")
print(f"Images found      : {found}")
print(f"Images missing    : {missing}")
print(f"Coverage          : {coverage:.1f}%")
if missing_paths:
    print("\nFirst 5 missing paths:")
    for p in missing_paths:
        print(f"  {p}")
