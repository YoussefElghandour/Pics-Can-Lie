"""
1.1 — extract_test_split.py
---------------------------
Extract MMFakeBench_test.zip into E:\\Pics Can Lie\\dataset\\MMFakeBench\\MMFakeBench_test\\
(skipped if the directory already contains the full set of images).

Then verify:
  - every sample in MMFakeBench_test.json has its image on disk,
  - fake / real distribution matches what we expect for an MMFakeBench training-style split
    (gt_answers == 'Fake' is positive, 'True' is negative — NOT 'False').

Run:
    python mmfakebench_training/extract_test_split.py
"""
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402


import json
import os
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
MMFB_ROOT  = Path(_os.path.join(str(_cfg.MMFB_ROOT)))
TEST_ZIP   = MMFB_ROOT / "MMFakeBench_test.zip"
TEST_DIR   = MMFB_ROOT / "MMFakeBench_test"
TEST_JSON  = MMFB_ROOT / "MMFakeBench_test.json"


def count_images(root: Path) -> int:
    """Walk root counting common image extensions."""
    if not root.exists():
        return 0
    n = 0
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                n += 1
    return n


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract zip to dest with a tqdm progress bar (skips members that already exist)."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for m in tqdm(members, desc=f"Extracting {zip_path.name}", unit="file"):
            target = dest / m
            if target.exists() and target.is_file():
                continue
            zf.extract(m, dest)


def main() -> None:
    if not TEST_JSON.exists():
        raise FileNotFoundError(f"Missing annotations: {TEST_JSON}")

    with open(TEST_JSON, encoding="utf-8") as f:
        ann = json.load(f)
    print(f"Annotations: {len(ann)} samples in {TEST_JSON.name}")

    n_existing = count_images(TEST_DIR)
    print(f"Images already on disk: {n_existing}")

    if n_existing < len(ann):
        if not TEST_ZIP.exists():
            raise FileNotFoundError(
                f"Test images incomplete ({n_existing}/{len(ann)}) "
                f"and zip not found at {TEST_ZIP}"
            )
        print(f"Extracting {TEST_ZIP.name} -> {TEST_DIR} ...")
        extract_zip(TEST_ZIP, TEST_DIR)
        n_existing = count_images(TEST_DIR)
        print(f"Images on disk after extraction: {n_existing}")
    else:
        print("Extraction skipped (images already present).")

    # ── Verify every annotated image exists ──────────────────────────────────
    missing = []
    for s in tqdm(ann, desc="Verifying image paths", unit="img"):
        # image_path is like '/real/bbc_test_500/bbc_test_0.png'  → join under TEST_DIR
        rel = s["image_path"].lstrip("/\\")
        if not (TEST_DIR / rel).exists():
            missing.append(rel)

    print(f"\nVerification result:")
    print(f"  Annotated samples : {len(ann)}")
    print(f"  Found on disk     : {len(ann) - len(missing)}")
    print(f"  Missing           : {len(missing)}")
    if missing[:5]:
        print(f"  First missing     : {missing[:5]}")

    # ── Distribution report (correct label convention: 'Fake' is positive) ───
    df = pd.DataFrame(ann)
    df["label"] = (df["gt_answers"] == "Fake").astype(int)
    n_fake = int(df["label"].sum())
    n_real = int(len(df) - n_fake)
    print(f"\nClass distribution (gt_answers convention):")
    print(f"  Fake (label=1) : {n_fake}  ({n_fake/len(df):.1%})")
    print(f"  True (label=0) : {n_real}  ({n_real/len(df):.1%})")

    print(f"\nfake_cls breakdown:")
    for k, v in Counter(df["fake_cls"]).most_common():
        print(f"  {k:<35} {v}")

    print(f"\nimage_source breakdown:")
    for k, v in Counter(df["image_source"]).most_common():
        print(f"  {k:<35} {v}")

    if missing:
        raise SystemExit(f"\nABORT: {len(missing)} annotated images missing on disk.")
    print("\nOK — extraction complete and verified.")


if __name__ == "__main__":
    main()
