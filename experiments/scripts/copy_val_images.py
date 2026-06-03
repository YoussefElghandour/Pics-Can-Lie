import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402
import json
import os
import shutil

VAL_LABELS  = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'merged_balanced', 'val.json')
VAL_META    = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'data', 'NewsClipPings', 'metadata', 'val.json')
SRC_ROOT    = _os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'origin')
DST_ROOT    = _os.path.join(str(_cfg.ROOT), 'kaggle_dataset_full', 'images')

with open(VAL_LABELS) as f:
    annotations = json.load(f)["annotations"]
with open(VAL_META) as f:
    metadata = json.load(f)

copied          = 0
missing_source  = 0
already_existed = 0

for i, ann in enumerate(annotations):
    key = str(ann["image_id"])
    if key not in metadata:
        missing_source += 1
        continue

    stripped = metadata[key]["image_path"].removeprefix("visual_news/")
    src      = os.path.join(SRC_ROOT, stripped)
    dst      = os.path.join(DST_ROOT, stripped)

    if os.path.exists(dst):
        already_existed += 1
        continue

    if not os.path.exists(src):
        missing_source += 1
        continue

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    copied += 1

    if copied % 500 == 0:
        print(f"  Copied {copied} files so far...")

total = copied + missing_source + already_existed
print(f"\nDone.")
print(f"  Total annotations : {len(annotations)}")
print(f"  Copied            : {copied}")
print(f"  Already existed   : {already_existed}")
print(f"  Missing from src  : {missing_source}")
