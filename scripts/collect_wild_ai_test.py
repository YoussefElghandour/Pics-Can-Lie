"""
collect_wild_ai_test.py — Phase 0 of the Ateeq wild-image sanity test.

Downloads photorealistic AI and real images from as many DISTINCT sources as are
reachable, saving them in NATURAL format/resolution (no re-encode, no resize) so
the processing-shortcut probe stays valid. Writes a manifest.csv logging the
source/generator of every image.

Reachable sources (after probing): AI = StyleGAN2 (thispersondoesnotexist) +
Midjourney v6 (HF datasets-server). Real = Pascal-VOC (Flickr), Imagenette
(ImageNet), Wikimedia Commons. Lexica is down; SD/SDXL HF datasets are gated.
"""
from __future__ import annotations
import csv, io, json, os, socket, sys, time, urllib.parse, urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

socket.setdefaulttimeout(30)
UA = {"User-Agent": "academic-research/1.0 (AI-detector audit; contact lab)"}
OUT = config.DATA_ROOT / "wild_ai_test"
AI_DIR, REAL_DIR = OUT / "ai", OUT / "real"
AI_DIR.mkdir(parents=True, exist_ok=True)
REAL_DIR.mkdir(parents=True, exist_ok=True)

EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
       "binary/octet-stream": ".jpg", "image/jpg": ".jpg"}
manifest: list[dict] = []


def fetch(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA))


def fetch_json(url: str):
    return json.load(fetch(url))


def save(raw: bytes, ct: str, dest_dir: Path, stem: str) -> Path | None:
    ext = EXT.get(ct.split(";")[0].strip(), ".jpg")
    # sniff PNG/JPEG magic if content-type is generic
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        ext = ".png"
    elif raw[:3] == b"\xff\xd8\xff":
        ext = ".jpg"
    p = dest_dir / f"{stem}{ext}"
    p.write_bytes(raw)
    return p


def hf_image_urls(dataset: str, config_name: str, split: str, n: int, want_label=None,
                  label_field="label"):
    """Yield image src URLs (optionally filtered by class label) from datasets-server."""
    ds = urllib.parse.quote(dataset, safe="")
    url = (f"https://datasets-server.huggingface.co/rows?dataset={ds}"
           f"&config={config_name}&split={split}&offset=0&length={min(n*3,100)}")
    d = fetch_json(url)
    out = []
    for row in d.get("rows", []):
        r = row["row"]
        if want_label is not None and r.get(label_field) != want_label:
            continue
        for k, v in r.items():
            if isinstance(v, dict) and "src" in v:
                out.append(v["src"]); break
    return out[:n]


# ── AI source 1: StyleGAN2 (thispersondoesnotexist) ──────────────────────────
def collect_stylegan(n=13):
    ok = 0
    for i in range(n * 2):
        if ok >= n:
            break
        try:
            r = fetch("https://thispersondoesnotexist.com/")
            raw = r.read()
            if len(raw) < 20000:
                continue
            p = save(raw, r.headers.get("Content-Type", "image/jpeg"), AI_DIR, f"stylegan_{ok:02d}")
            manifest.append({"path": str(p), "class": "ai", "source": "thispersondoesnotexist",
                             "generator_or_outlet": "StyleGAN2"})
            ok += 1
            time.sleep(1.0)  # give the server a fresh seed
        except Exception as e:
            print(f"  stylegan {i}: {type(e).__name__}")
            time.sleep(1.5)
    print(f"StyleGAN2: {ok} images")
    return ok


# ── AI source 2: Midjourney v6 (HF) ──────────────────────────────────────────
def collect_midjourney(n=13):
    ok = 0
    try:
        urls = hf_image_urls("brivangl/midjourney-v6-llava", "default", "train", n)
    except Exception as e:
        print(f"  midjourney list FAIL: {e}"); return 0
    for j, u in enumerate(urls):
        try:
            r = fetch(u); raw = r.read()
            if len(raw) < 20000:
                continue
            p = save(raw, r.headers.get("Content-Type", "image/jpeg"), AI_DIR, f"midjourney_{ok:02d}")
            manifest.append({"path": str(p), "class": "ai", "source": "hf:brivangl/midjourney-v6-llava",
                             "generator_or_outlet": "Midjourney_v6"})
            ok += 1
        except Exception as e:
            print(f"  midjourney {j}: {type(e).__name__}")
    print(f"Midjourney v6: {ok} images")
    return ok


# ── Real sources: Pascal-VOC, Imagenette (HF) + Wikimedia Commons ────────────
def collect_hf_real(dataset, cfg, label, outlet, prefix, n=9):
    ok = 0
    try:
        urls = hf_image_urls(dataset, cfg, "train", n)
    except Exception as e:
        print(f"  {outlet} list FAIL: {e}"); return 0
    for j, u in enumerate(urls):
        try:
            r = fetch(u); raw = r.read()
            if len(raw) < 15000:
                continue
            p = save(raw, r.headers.get("Content-Type", "image/jpeg"), REAL_DIR, f"{prefix}_{ok:02d}")
            manifest.append({"path": str(p), "class": "real", "source": f"hf:{dataset}",
                             "generator_or_outlet": outlet})
            ok += 1
        except Exception as e:
            print(f"  {outlet} {j}: {type(e).__name__}")
    print(f"{outlet}: {ok} images")
    return ok


def collect_wikimedia(n=9):
    ok = 0
    terms = ["portrait photograph", "street scene", "landscape photograph",
             "building exterior", "people candid"]
    for term in terms:
        if ok >= n:
            break
        try:
            q = urllib.parse.quote(f"filetype:bitmap {term}")
            api = ("https://commons.wikimedia.org/w/api.php?action=query&generator=search"
                   f"&gsrsearch={q}&gsrnamespace=6&gsrlimit=3&prop=imageinfo"
                   "&iiprop=url|size|mime&format=json")
            d = fetch_json(api)
            pages = (d.get("query", {}) or {}).get("pages", {})
            for _, pg in pages.items():
                if ok >= n:
                    break
                ii = pg.get("imageinfo", [{}])[0]
                if ii.get("mime") != "image/jpeg":
                    continue
                url = ii.get("url")
                if not url:
                    continue
                try:
                    r = fetch(url); raw = r.read()
                    if len(raw) < 15000 or len(raw) > 25_000_000:
                        continue
                    p = save(raw, "image/jpeg", REAL_DIR, f"wikimedia_{ok:02d}")
                    manifest.append({"path": str(p), "class": "real", "source": "wikimedia_commons",
                                     "generator_or_outlet": "WikimediaCommons"})
                    ok += 1
                except Exception as e:
                    print(f"  wikimedia dl: {type(e).__name__}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  wikimedia search '{term}': {type(e).__name__}")
    print(f"WikimediaCommons: {ok} images")
    return ok


def main():
    print(f"Collecting into {OUT}\n--- AI ---")
    collect_stylegan(13)
    collect_midjourney(13)
    print("--- REAL ---")
    collect_hf_real("nateraw/pascal-voc-2012", "default", None, "PascalVOC(Flickr)", "pascal", 9)
    collect_hf_real("frgfm/imagenette", "full_size", None, "Imagenette(ImageNet)", "imagenette", 9)
    collect_wikimedia(9)

    man_path = OUT / "manifest.csv"
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class", "source", "generator_or_outlet"])
        w.writeheader(); w.writerows(manifest)

    # report
    from collections import Counter
    ai = [m for m in manifest if m["class"] == "ai"]
    real = [m for m in manifest if m["class"] == "real"]
    ai_src = Counter(m["generator_or_outlet"] for m in ai)
    real_src = Counter(m["generator_or_outlet"] for m in real)
    print("\n===== PHASE 0 REPORT =====")
    print(f"AI:   n={len(ai)}  distinct sources={len(ai_src)}  -> {dict(ai_src)}")
    print(f"Real: n={len(real)}  distinct sources={len(real_src)}  -> {dict(real_src)}")
    print(f"manifest -> {man_path}")
    if len(ai_src) < 3 or len(real_src) < 3:
        print("\n*** WARNING: a class has < 3 distinct sources -> pool-homogeneity confound risk. ***")
        print("*** STOP: manual diversity injection recommended before Phase 1 scoring.        ***")


if __name__ == "__main__":
    main()
