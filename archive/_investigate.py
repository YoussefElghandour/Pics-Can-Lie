"""Two checks:
   1. Verify val_sample_ids.csv label convention (1=FAKE vs 1=REAL)
   2. Investigate why s4/s5/s6 are zero for most FAKE samples in evidence join.
"""
import os, json, csv
import numpy as np
import pandas as pd
from collections import Counter

ROOT       = r"D:\Pics Can Lie"
SAMPLE_IDS = rf"{ROOT}\clip_finetuned_v2\val_features\val_sample_ids.csv"
EVID       = rf"{ROOT}\evidence_clip_scores.csv"
META       = rf"{ROOT}\dataset\data\NewsClipPings\metadata\val.json"
ANN        = rf"{ROOT}\dataset\data\NewsClipPings\merged_balanced\val.json"

print("=" * 70)
print("CHECK 1: val_sample_ids.csv label vs annotations.falsified")
print("=" * 70)

with open(ANN, "r", encoding="utf-8") as f:
    ann = json.load(f)

# annotations: same id can appear in two annotation rows (one REAL, one FAKE).
# A pair is uniquely identified by (id, image_id).  For REAL pairs id==image_id.
# Build a lookup keyed by id → list of {image_id, falsified}.
ann_lookup = {}
for a in ann["annotations"]:
    ann_lookup.setdefault(int(a["id"]), []).append(
        (int(a["image_id"]), bool(a["falsified"]))
    )

# Load CSV
csv_rows = []
with open(SAMPLE_IDS, "r", encoding="utf-8", newline="") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        csv_rows.append({"idx": int(r["idx"]), "id": int(r["id"]), "label": int(r["label"])})

# CSV label vs annotations: for each id in CSV, look up annotation rows.
# Count cases where the CSV says label=1 AND annotation has falsified=True for that id.
n_csv_1   = sum(1 for r in csv_rows if r["label"] == 1)
n_csv_0   = sum(1 for r in csv_rows if r["label"] == 0)
print(f"  CSV label=1 count: {n_csv_1}")
print(f"  CSV label=0 count: {n_csv_0}")

# Among CSV label=1 rows, how many ids have AT LEAST ONE falsified=True annotation?
agree_1_fake = sum(1 for r in csv_rows if r["label"] == 1
                   and any(f for _, f in ann_lookup.get(r["id"], [])))
agree_0_real = sum(1 for r in csv_rows if r["label"] == 0
                   and any(not f for _, f in ann_lookup.get(r["id"], [])))
print(f"  CSV label=1 ids that have falsified=True in annotations: {agree_1_fake}/{n_csv_1}")
print(f"  CSV label=0 ids that have falsified=False in annotations: {agree_0_real}/{n_csv_0}")
# Inverse check (suggests inverted convention)
agree_1_real = sum(1 for r in csv_rows if r["label"] == 1
                   and any(not f for _, f in ann_lookup.get(r["id"], [])))
agree_0_fake = sum(1 for r in csv_rows if r["label"] == 0
                   and any(f for _, f in ann_lookup.get(r["id"], [])))
print(f"  (inverse) CSV label=1 ids with falsified=False: {agree_1_real}/{n_csv_1}")
print(f"  (inverse) CSV label=0 ids with falsified=True : {agree_0_fake}/{n_csv_0}")

# Look at sample C specifically (id=1573659)
print("\n  Sample C id=1573659 annotation rows:")
for img_id, f in ann_lookup.get(1573659, []):
    print(f"    image_id={img_id}  falsified={f}")
print("  CSV row for id=1573659:")
for r in csv_rows:
    if r["id"] == 1573659:
        print(f"    {r}")

# Same for 404539 and 308192
for sid in (404539, 308192):
    print(f"\n  id={sid} annotation rows:")
    for img_id, f in ann_lookup.get(sid, []):
        print(f"    image_id={img_id}  falsified={f}")
    print(f"  CSV row for id={sid}:")
    for r in csv_rows:
        if r["id"] == sid:
            print(f"    {r}")


print("\n" + "=" * 70)
print("CHECK 2: evidence CSV — why s4/s5/s6 == 0 for FAKE samples")
print("=" * 70)

ev = pd.read_csv(EVID, dtype={"id": str})
print(f"  rows: {len(ev)}  unique ids: {ev['id'].nunique()}")
print(f"  per-column zero rate:")
for col in ["s2", "s3", "s4", "s5", "s6"]:
    z = (ev[col] == 0).mean()
    print(f"    {col}: zero={z*100:.1f}%   min={ev[col].min():.4f}  max={ev[col].max():.4f}  mean={ev[col].mean():.4f}")

# Cross with CSV labels: are s4/s5/s6 zero specifically for label=1 (FAKE) rows?
ev["id_int"] = ev["id"].astype(int)
csv_df = pd.DataFrame(csv_rows)
merged = csv_df.merge(ev, left_on="id", right_on="id_int", how="left")
print(f"\n  After merging CSV (5000 rows) with evidence on id:")
print(f"    rows with NO evidence row (NaN): {merged['s2'].isna().sum()}")
print(f"  Zero-rate per column, split by label:")
for col in ["s2", "s3", "s4", "s5", "s6"]:
    z1 = ((merged[merged["label"] == 1][col] == 0) | merged[merged["label"] == 1][col].isna()).mean()
    z0 = ((merged[merged["label"] == 0][col] == 0) | merged[merged["label"] == 0][col].isna()).mean()
    print(f"    {col}: label=1 zero/NaN {z1*100:.1f}%   label=0 zero/NaN {z0*100:.1f}%")

# Was evidence computed only for one direction? Check if all evidence ids are
# from a particular subset (e.g. only article ids that appear as "real" pair).
print(f"\n  Are all evidence-CSV ids present in val annotations as id?")
ann_ids = set(int(a["id"]) for a in ann["annotations"])
ev_ids  = set(int(i) for i in ev["id"])
print(f"    evidence ids ⊆ annotation ids: {ev_ids.issubset(ann_ids)}")
print(f"    evidence ids in annotations: {len(ev_ids & ann_ids)}/{len(ev_ids)}")

# Inspect a few rows where s4=s5=s6=0
zero_rows = ev[(ev["s4"] == 0) & (ev["s5"] == 0) & (ev["s6"] == 0)]
nonzero_rows = ev[(ev["s4"] != 0) & (ev["s5"] != 0) & (ev["s6"] != 0)]
print(f"\n  rows with s4=s5=s6=0: {len(zero_rows)} ({100*len(zero_rows)/len(ev):.1f}%)")
print(f"  rows with s4,s5,s6 all nonzero: {len(nonzero_rows)} ({100*len(nonzero_rows)/len(ev):.1f}%)")
print(f"  example zero-row:    {zero_rows.iloc[0].to_dict()}")
print(f"  example nonzero-row: {nonzero_rows.iloc[0].to_dict()}")
