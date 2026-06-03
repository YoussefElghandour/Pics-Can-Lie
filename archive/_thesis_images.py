import os, json, shutil
from PIL import Image

ROOT       = r"D:\Pics Can Lie"
META       = rf"{ROOT}\dataset\data\NewsClipPings\metadata\val.json"
IMG_ROOT   = rf"{ROOT}\dataset\origin\origin"
OUT_DIR    = rf"{ROOT}\thesis_examples"
os.makedirs(OUT_DIR, exist_ok=True)

with open(META, "r", encoding="utf-8") as f:
    meta = json.load(f)

# (article_id, image_id, label, tag)
# Note: for REAL pairs image_id == article_id by construction.
EXAMPLES = [
    (404539,  75104,    "FAKE", "A_wiki_veto"),
    (308192,  1238256,  "FAKE", "B_clip_dominant"),
    (1573659, 1573659,  "REAL", "C_correct_real"),   # REAL → article's own image
]

def strip_prefix(p):
    p = p.replace("\\", "/")
    for pref in ("visual_news/origin/", "origin/"):
        if p.startswith(pref):
            return p[len(pref):]
    return p

for art_id, img_id, label, tag in EXAMPLES:
    cap_entry = meta[str(art_id)]
    img_entry = meta[str(img_id)]   # for REAL pairs this == cap_entry
    rel = strip_prefix(img_entry["image_path"])
    src = os.path.join(IMG_ROOT, rel.replace("/", os.sep))
    print(f"\n=== {tag}  (article_id={art_id}, image_id={img_id}, label={label}) ===")
    print(f"  caption       : {cap_entry['caption']}")
    print(f"  source        : {cap_entry.get('source')}  topic: {cap_entry.get('topic')}")
    print(f"  image path    : {src}")
    print(f"  exists?       : {os.path.exists(src)}")
    if os.path.exists(src):
        try:
            im = Image.open(src)
            print(f"  image size    : {im.size}  mode: {im.mode}")
            dst = os.path.join(OUT_DIR, f"{tag}__art{art_id}_img{img_id}{os.path.splitext(src)[1].lower()}")
            shutil.copy2(src, dst)
            print(f"  copied to     : {dst}")
        except Exception as e:
            print(f"  IMAGE ERROR: {e}")

    if art_id != img_id:
        # Show the OTHER image (the "real" pair) for comparison
        real_rel = strip_prefix(cap_entry["image_path"])
        real_src = os.path.join(IMG_ROOT, real_rel.replace("/", os.sep))
        print(f"  (article's original image, for context): {real_src}  exists={os.path.exists(real_src)}")
        if os.path.exists(real_src):
            dst = os.path.join(OUT_DIR, f"{tag}__art{art_id}_originalimg{os.path.splitext(real_src)[1].lower()}")
            shutil.copy2(real_src, dst)
            print(f"  copied to     : {dst}")

print(f"\nAll images saved under: {OUT_DIR}")
