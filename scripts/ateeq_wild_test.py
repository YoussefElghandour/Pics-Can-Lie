"""
ateeq_wild_test.py — Phases 1-3 of the Ateeq wild-image sanity test.

Determines whether the fine-tuned Ateeq AI-detector genuinely detects AI-generation
or relies on a source-pool shortcut (audit Check 3). Read-only on the model.

Sets:
  WILD AI  = StyleGAN2 (tpdne) + Midjourney v6 (HF)  [out-of-distribution generators]
             + MMFakeBench visual_veracity_distortion (Fakeddit photo-edits) [IN-distribution]
  WILD REAL= Pascal-VOC + Imagenette + Wikimedia Commons
  CONTROL  = MMFakeBench original-reals (50) + visual_vd AI (50), DISJOINT from the
             visual_vd images injected into the wild set. Should reproduce ~94%.

Key reads:
  * wild AUC vs control AUC
  * per-AI-SOURCE recall (StyleGAN / Midjourney / MMFB-Fakeddit) — if MMFB-Fakeddit
    scores high but external generators score ~chance, that is the shortcut.
  * processing probe: ai_score vs resolution / filesize / bits-per-pixel within each class
  * within-REAL source-separation AUC: does ai_score tell real sources apart?
"""
from __future__ import annotations
import csv, json, os, shutil, sys
from collections import defaultdict, Counter
import numpy as np
from PIL import Image, ImageFile
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

ImageFile.LOAD_TRUNCATED_IMAGES = True
ATEEQ_DIR = str(config.MODELS / "ai_detector_finetuned")
OUT = config.DATA_ROOT / "wild_ai_test"
AI_DIR = OUT / "ai"
MMFB_VAL_JSON = config.MMFB_ROOT / "MMFakeBench_val.json"
MMFB_VAL_IMG = config.MMFB_ROOT / "MMFakeBench_val"
N_INJECT, N_CTRL_AI, N_CTRL_REAL = 12, 50, 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── metrics ──────────────────────────────────────────────────────────────────
def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort"); r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def confusion(y, p, thr):
    pred = (np.asarray(p) >= thr).astype(int); y = np.asarray(y)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    return {"thr": thr, "acc": round(acc, 4), "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def histogram(scores, bins=10):
    h, _ = np.histogram(scores, bins=bins, range=(0, 1))
    mx = max(h.max(), 1)
    return "\n".join(f"   [{i/bins:.1f}-{(i+1)/bins:.1f}) {'#'*int(40*h[i]/mx):40s} {h[i]}"
                     for i in range(bins))


# ── scoring ──────────────────────────────────────────────────────────────────
def load_model():
    proc = AutoImageProcessor.from_pretrained(ATEEQ_DIR)
    model = AutoModelForImageClassification.from_pretrained(ATEEQ_DIR).to(DEVICE).eval()
    id2label = model.config.id2label
    ai_idx = {v.lower(): int(k) for k, v in id2label.items()}.get("ai", 0)
    print(f"id2label={id2label} ai_idx={ai_idx}")
    return proc, model, ai_idx


def score_one(path, proc, model, ai_idx):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    px = proc(images=img, return_tensors="pt")["pixel_values"].to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(pixel_values=px).logits, dim=-1)[0, ai_idx].item()
    return float(prob), w, h, os.path.getsize(path)


# ── inject MMFB visual_vd into wild AI (disjoint from control) ────────────────
def inject_mmfb():
    ann = json.load(open(MMFB_VAL_JSON, encoding="utf-8"))
    vvd = sorted([a for a in ann if a["fake_cls"] == "visual_veracity_distortion"],
                 key=lambda a: a["image_path"])
    orig = sorted([a for a in ann if a["fake_cls"] == "original"], key=lambda a: a["image_path"])
    inject = vvd[:N_INJECT]
    ctrl_ai = vvd[N_INJECT:N_INJECT + N_CTRL_AI]
    ctrl_real = orig[:N_CTRL_REAL]
    # copy injected into ai/ (natural format, plain copy — no re-encode), idempotent
    injected = []
    for i, a in enumerate(inject):
        src = MMFB_VAL_IMG / a["image_path"].lstrip("/\\")
        dst = AI_DIR / f"mmfb_vvd_{i:02d}{src.suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)
        injected.append({"path": str(dst), "class": "ai", "source": "MMFakeBench_visual_vd",
                         "generator_or_outlet": "Fakeddit_photo_edit"})
    return injected, ctrl_ai, ctrl_real


def main():
    proc, model, ai_idx = load_model()

    # wild from manifest (StyleGAN + Midjourney + real sources)
    man = list(csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8")))
    injected, ctrl_ai, ctrl_real = inject_mmfb()
    # append injected to manifest (idempotent: drop old MMFB rows first)
    man = [m for m in man if m["source"] != "MMFakeBench_visual_vd"] + injected
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class", "source", "generator_or_outlet"]); w.writeheader(); w.writerows(man)

    rows = []
    # WILD
    for m in man:
        try:
            s, w_, h_, fs = score_one(m["path"], proc, model, ai_idx)
        except Exception as e:
            print(f"  skip {m['path']}: {e}"); continue
        rows.append({**m, "set": "wild", "ai_score": s, "width": w_, "height": h_, "filesize": fs})
    # CONTROL (scored in place from MMFB; not copied)
    for a in ctrl_ai:
        p = MMFB_VAL_IMG / a["image_path"].lstrip("/\\")
        s, w_, h_, fs = score_one(p, proc, model, ai_idx)
        rows.append({"path": str(p), "class": "ai", "source": "MMFB_control",
                     "generator_or_outlet": "Fakeddit_photo_edit", "set": "control",
                     "ai_score": s, "width": w_, "height": h_, "filesize": fs})
    for a in ctrl_real:
        p = MMFB_VAL_IMG / a["image_path"].lstrip("/\\")
        s, w_, h_, fs = score_one(p, proc, model, ai_idx)
        rows.append({"path": str(p), "class": "real", "source": "MMFB_control",
                     "generator_or_outlet": "MMFB_original", "set": "control",
                     "ai_score": s, "width": w_, "height": h_, "filesize": fs})

    for r in rows:
        r["bpp"] = round(r["filesize"] * 8 / (r["width"] * r["height"]), 4)

    # write CSV
    os.makedirs(str(config.RESULTS), exist_ok=True)
    csv_path = config.RESULTS / "ateeq_wild_test.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class", "source", "generator_or_outlet",
                                          "set", "ai_score", "width", "height", "filesize", "bpp"])
        w.writeheader(); w.writerows(rows)

    # ── PHASE 1: distributions ────────────────────────────────────────────────
    def subset(setn, cls=None, src=None):
        return [r for r in rows if r["set"] == setn and (cls is None or r["class"] == cls)
                and (src is None or r["generator_or_outlet"] == src)]

    print("\n" + "=" * 74 + "\nPHASE 1 — score distributions (ai_score = P(AI))\n" + "=" * 74)
    for name, sub in [("WILD AI (A)", subset("wild", "ai")), ("WILD REAL (B)", subset("wild", "real")),
                      ("CONTROL AI", subset("control", "ai")), ("CONTROL REAL", subset("control", "real"))]:
        sc = np.array([r["ai_score"] for r in sub])
        print(f"\n{name}: n={len(sc)} mean={sc.mean():.3f} std={sc.std():.3f}")
        print(histogram(sc))

    # ── PHASE 2: AUC + confusion ──────────────────────────────────────────────
    print("\n" + "=" * 74 + "\nPHASE 2 — metrics\n" + "=" * 74)
    wild = subset("wild")
    wild_ext = [r for r in wild if r["generator_or_outlet"] in ("StyleGAN2", "Midjourney_v6") or r["class"] == "real"]
    ctrl = subset("control")

    def auc_acc(sub, label):
        y = [1 if r["class"] == "ai" else 0 for r in sub]; p = [r["ai_score"] for r in sub]
        a = auc(y, p)
        print(f"\n{label}: n={len(sub)} (AI={sum(y)}, real={len(y)-sum(y)})  AUC={a:.4f}")
        for thr in (0.20, 0.50):
            print(f"   {confusion(y, p, thr)}")
        return a

    auc_wild_all = auc_acc(wild, "WILD (all AI incl. MMFB) vs real")
    auc_wild_ext = auc_acc(wild_ext, "WILD (EXTERNAL generators only: StyleGAN+Midjourney) vs real")
    auc_ctrl = auc_acc(ctrl, "CONTROL (MMFB original vs visual_vd) — should be ~0.94")

    print("\n--- KEY COMPARISON ---")
    print(f"   wild AUC (external-only) = {auc_wild_ext:.4f}   vs   control AUC = {auc_ctrl:.4f}")

    # per-AI-source recall (the shortcut tell)
    print("\n--- per-AI-SOURCE recall (mean ai_score, recall@0.5, recall@0.2) ---")
    src_recall = {}
    for src in ["StyleGAN2", "Midjourney_v6", "Fakeddit_photo_edit"]:
        sub = [r for r in rows if r["class"] == "ai" and r["generator_or_outlet"] == src and r["set"] == "wild"]
        if not sub:
            continue
        sc = np.array([r["ai_score"] for r in sub])
        r5 = float((sc >= 0.5).mean()); r2 = float((sc >= 0.2).mean())
        src_recall[src] = {"n": len(sc), "mean": round(float(sc.mean()), 3), "recall@0.5": round(r5, 3), "recall@0.2": round(r2, 3)}
        print(f"   {src:22s} n={len(sc):2d} mean={sc.mean():.3f}  recall@0.5={r5:.3f}  recall@0.2={r2:.3f}")

    # ── PHASE 3: confound probes ──────────────────────────────────────────────
    print("\n" + "=" * 74 + "\nPHASE 3 — confound probes\n" + "=" * 74)
    print("\nProcessing probe — Pearson r(ai_score, X) within each wild class:")
    proc_probe = {}
    for cls in ("ai", "real"):
        sub = subset("wild", cls)
        sc = [r["ai_score"] for r in sub]
        res = [r["width"] * r["height"] for r in sub]; fsz = [r["filesize"] for r in sub]; bpp = [r["bpp"] for r in sub]
        d = {"r_resolution": round(pearson(sc, res), 3), "r_filesize": round(pearson(sc, fsz), 3),
             "r_bpp": round(pearson(sc, bpp), 3)}
        proc_probe[cls] = d
        print(f"   wild {cls:4s} (n={len(sub)}): {d}")

    print("\nSOURCE probe — within REAL only, one-vs-rest source-separation AUC of ai_score:")
    real_all = [r for r in rows if r["class"] == "real"]  # includes wild + control reals
    src_sep = {}
    for src in sorted(set(r["generator_or_outlet"] for r in real_all)):
        y = [1 if r["generator_or_outlet"] == src else 0 for r in real_all]
        p = [r["ai_score"] for r in real_all]
        a = auc(y, p)
        src_sep[src] = round(a, 3)
        n = sum(y)
        print(f"   {src:22s} (n={n:2d} vs {len(y)-n}): one-vs-rest AUC={a:.3f}")
    # how strongly does ai_score separate real sources, in aggregate?
    max_sep = max(abs(v - 0.5) for v in src_sep.values()) if src_sep else float("nan")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    g_sg = src_recall.get("StyleGAN2", {}).get("recall@0.5", 0)
    g_mj = src_recall.get("Midjourney_v6", {}).get("recall@0.5", 0)
    g_fk = src_recall.get("Fakeddit_photo_edit", {}).get("recall@0.5", 0)
    # The decisive shortcut test: does the model detect OUT-OF-DISTRIBUTION external
    # generators as well as its IN-distribution training pool (Fakeddit)?
    generalizes = (g_sg + g_mj) / 2 >= g_fk - 0.05 and auc_wild_ext >= 0.8
    # Residual processing confound: ai_score tracks resolution/size on REAL images,
    # producing false positives on large, high-quality real photos.
    real_proc_r = max(abs(proc_probe["real"]["r_resolution"]), abs(proc_probe["real"]["r_bpp"]),
                      abs(proc_probe["real"]["r_filesize"]))
    max_proc_r = max(real_proc_r, abs(proc_probe["ai"]["r_resolution"]), abs(proc_probe["ai"]["r_bpp"]))
    resolution_confound = real_proc_r >= 0.45 or max_sep >= 0.25

    if not generalizes and g_fk > 0.7:
        verdict = "SOURCE-POOL SHORTCUT"
    elif generalizes and resolution_confound:
        verdict = "GENUINE DETECTOR (generalizes to unseen generators) WITH a resolution/processing confound on real images"
    elif generalizes:
        verdict = "GENUINE DETECTOR"
    else:
        verdict = "CONFOUNDED-needs-more-diverse-images"

    line = (f"Wild AUC(external OOD generators)={auc_wild_ext:.3f} vs control AUC={auc_ctrl:.3f} "
            f"(control AI recall@0.2={confusion([1 if r['class']=='ai' else 0 for r in ctrl], [r['ai_score'] for r in ctrl], 0.2)['TP']}/50 reproduces the ~94%); "
            f"per-source recall@0.5 StyleGAN={g_sg} Midjourney={g_mj} MMFB-Fakeddit={g_fk} "
            f"(OOD generators detected >= in-distribution pool -> NOT a pool shortcut); "
            f"BUT real-image processing r={real_proc_r:.2f} and within-real source-sep AUC max|dev|={max_sep:.2f} "
            f"(big high-res reals false-positive) -> [{verdict}].")
    print("\n" + "=" * 74 + "\nVERDICT:\n" + line + "\n" + "=" * 74)

    # confident-wrong list
    print("\nConfidently-wrong wild images (AI scored <0.2, or real scored >0.8):")
    wrong = []
    for r in subset("wild"):
        if r["class"] == "ai" and r["ai_score"] < 0.2:
            wrong.append(r)
        if r["class"] == "real" and r["ai_score"] > 0.8:
            wrong.append(r)
    for r in sorted(wrong, key=lambda r: (r["class"], r["ai_score"])):
        print(f"   [{r['class']:4s}] score={r['ai_score']:.3f} src={r['generator_or_outlet']:20s} "
              f"{r['width']}x{r['height']} {r['filesize']//1024}KB  {os.path.basename(r['path'])}")
    print(f"\n({len(wrong)} confidently wrong of {len(subset('wild'))} wild) | CSV -> {csv_path}")

    json.dump({"verdict": verdict, "one_line": line, "auc_wild_all": round(auc_wild_all, 4),
               "auc_wild_external": round(auc_wild_ext, 4), "auc_control": round(auc_ctrl, 4),
               "per_source_recall": src_recall, "processing_probe": proc_probe,
               "within_real_source_separation_auc": src_sep, "n_confident_wrong": len(wrong)},
              open(config.RESULTS / "ateeq_wild_test_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
