"""
run_test_inference_fixed.py  (Prompt A — corrected test path)

Fixes the two coupled bugs the audit found in the original run_test_inference:
  (a) threshold was swept over the TEST labels (data snooping -> fake 86.48%);
  (b) the StandardScaler from training was never applied and scalars were fed
      in the wrong order, so AITR saw out-of-distribution inputs squashed to ~0.02.

This version:
  * Recomputes CLIP v2 clip_prob / clip_sim and 768-d embeddings on the test images.
  * Assembles the 9 scalars in the TRAINING order
        [clip_prob, clip_sim, deberta, s2, s3, s4, s5, s6, wiki].
    The 7 evidence/NLI scalars do not exist for the test set (known limitation:
    test has no retrieved evidence), so they are imputed with the TRAINING means
    and then standardized through the persisted scaler -> they land near 0.
  * Applies fusion_aitr/scalar_scaler.joblib (fit on the TRAIN split only).
  * Applies fusion_aitr/frozen_threshold.json — a threshold frozen on TRAIN.
    No argmax over test labels anywhere.
  * Fixes the source lookup: metadata[str(id)]['source'] first, integer
    source_dataset only as fallback; falsified entries take the image from image_id.
  * Reports AUC (primary), accuracy at the frozen threshold, per-source accuracy,
    and 1000-resample 95% bootstrap CIs.

Run on the 4060:  python run_test_inference_fixed.py
"""

from __future__ import annotations

import json
import os

import clip
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import config

TEST_ANN  = str(config.TEST_ANN)
TEST_META = str(config.TEST_META)
IMAGES_ROOT = str(config.IMAGES_ROOT)
CLIP_CKPT = str(config.CLIP_CKPT)
AITR_CKPT = str(config.AITR_CKPT)
SCALER    = str(config.SCALER_PATH)
THRESH    = str(config.THRESH_PATH)
OUT_JSON  = str(config.RESULTS / "test_set_results" / "test_results_fixed.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
SCALAR_ORDER = ["clip_prob", "clip_sim", "deberta", "s2", "s3", "s4", "s5", "s6", "wiki"]


class CLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        dim = 768
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2 + 1, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1))

    def forward(self, images, input_ids):
        img_f = F.normalize(self.clip.encode_image(images), dim=-1)
        txt_f = F.normalize(self.clip.encode_text(input_ids), dim=-1)
        cos = (img_f * txt_f).sum(dim=-1, keepdim=True)
        x = torch.cat([img_f, txt_f, cos], dim=-1)
        return torch.sigmoid(self.classifier(x)).squeeze(-1), img_f, txt_f


class AITR(nn.Module):
    def __init__(self, embed_dim=768, scalar_dim=9, num_heads=8, num_layers=2,
                 dropout=0.3, hidden_dim=256):
        super().__init__()
        self.scalar_proj = nn.Sequential(nn.Linear(scalar_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.type_embedding = nn.Embedding(5, embed_dim)
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                         dim_feedforward=embed_dim * 2, dropout=dropout,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, hidden_dim),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, img_emb, txt_emb, scalar):
        B = img_emb.size(0)
        tok = torch.stack([img_emb, txt_emb, img_emb * txt_emb, img_emb - txt_emb,
                           self.scalar_proj(scalar)], dim=1)
        tid = torch.arange(5, device=tok.device).unsqueeze(0).expand(B, -1)
        tok = tok + self.type_embedding(tid)
        out = self.transformer(torch.cat([self.cls_token.expand(B, -1, -1), tok], dim=1))
        return torch.sigmoid(self.classifier(out[:, 0])).squeeze(-1)


def resolve_image_path(rel):
    rel = rel.replace("\\", "/")
    for pre in ("visual_news/origin/", "origin/"):
        if rel.startswith(pre):
            rel = rel[len(pre):]; break
    return os.path.join(IMAGES_ROOT, rel.replace("/", os.sep))


def build_samples():
    ann = json.load(open(TEST_ANN, encoding="utf-8"))
    meta = json.load(open(TEST_META, encoding="utf-8"))
    out, skipped = [], 0
    for a in ann["annotations"]:
        cap_key, img_key = str(a["id"]), str(a["image_id"])
        if cap_key not in meta or img_key not in meta:
            skipped += 1; continue
        cap, imeta = meta[cap_key], meta[img_key]
        img_path = resolve_image_path(imeta["image_path"])  # falsified -> from image_id
        if not os.path.exists(img_path):
            skipped += 1; continue
        # FIX: metadata source first; integer source_dataset only as fallback.
        source = cap.get("source") or a.get("source_dataset") or "unknown"
        out.append({"id": a["id"], "image_id": a["image_id"], "falsified": bool(a["falsified"]),
                    "source": str(source), "caption": cap["caption"], "image_path": img_path})
    print(f"Loaded {len(out)} samples ({skipped} skipped).")
    return out


class TestDS(Dataset):
    def __init__(self, samples, preprocess):
        self.s = samples; self.pp = preprocess
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s = self.s[i]
        return self.pp(Image.open(s["image_path"]).convert("RGB")), clip.tokenize([s["caption"]], truncate=True)[0], i


def auc(y, p):
    order = np.argsort(p); r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_ci(y, p, thr, n=1000):
    rng = np.random.default_rng(42); idx = np.arange(len(y)); accs, aucs = [], []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2: continue
        accs.append(((p[b] >= thr).astype(int) == y[b]).mean()); aucs.append(auc(y[b], p[b]))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return q(accs), q(aucs)


def main():
    samples = build_samples()
    bundle = joblib.load(SCALER)
    scaler, order, train_means = bundle["scaler"], bundle["order"], np.array(bundle["train_means"])
    assert order == SCALAR_ORDER, f"scaler order mismatch: {order}"
    thr = json.load(open(THRESH))["frozen_threshold"]
    print(f"Loaded scaler + frozen threshold {thr:.4f}")

    clip_base, preprocess = clip.load("ViT-L/14", device=DEVICE, jit=False)
    clf = CLIPClassifier(clip_base.float()).to(DEVICE)
    clf.load_state_dict(torch.load(CLIP_CKPT, map_location=DEVICE)["model_state"]); clf.eval()

    aitr = AITR().to(DEVICE)
    st = torch.load(AITR_CKPT, map_location=DEVICE)
    aitr.load_state_dict(st["state_dict"] if "state_dict" in st else st); aitr.eval()

    loader = DataLoader(TestDS(samples, preprocess), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    results = []
    with torch.no_grad():
        for imgs, toks, idxs in loader:
            imgs, toks = imgs.to(DEVICE), toks.to(DEVICE)
            clip_prob, img_f, txt_f = clf(imgs, toks)
            clip_sim = (img_f * txt_f).sum(-1)
            B = imgs.size(0)
            # raw scalars: real clip_prob/sim; evidence/NLI imputed with TRAIN means
            raw = np.tile(train_means, (B, 1)).astype(np.float32)
            raw[:, 0] = clip_prob.cpu().numpy(); raw[:, 1] = clip_sim.cpu().numpy()
            scaled = torch.tensor(scaler.transform(raw), dtype=torch.float32, device=DEVICE)
            fused = aitr(img_f, txt_f, scaled)
            for j in range(B):
                s = samples[int(idxs[j])]; p = float(fused[j])
                results.append({**{k: s[k] for k in ("id", "image_id", "source")},
                                "label": s["falsified"], "clip_prob": float(clip_prob[j]),
                                "clip_sim": float(clip_sim[j]), "fused_prob": p,
                                "prediction": bool(p >= thr), "correct": bool(p >= thr) == s["falsified"]})

    y = np.array([1 if r["label"] else 0 for r in results])
    p = np.array([r["fused_prob"] for r in results])
    acc = float(((p >= thr).astype(int) == y).mean()); a = float(auc(y, p))
    (acc_lo, acc_hi), (auc_lo, auc_hi) = bootstrap_ci(y, p, thr)

    per_source = {}
    for r in results:
        d = per_source.setdefault(r["source"], {"total": 0, "correct": 0})
        d["total"] += 1; d["correct"] += int(r["correct"])
    for s, d in per_source.items():
        d["accuracy"] = d["correct"] / d["total"]

    summary = {"n": len(results), "frozen_threshold": thr,
               "AUC": a, "AUC_95ci": [auc_lo, auc_hi],
               "accuracy": acc, "accuracy_95ci": [acc_lo, acc_hi],
               "fused_prob_stats": {"min": float(p.min()), "max": float(p.max()),
                                    "mean": float(p.mean()), "std": float(p.std())},
               "per_source": per_source, "protocol": "threshold frozen on TRAIN; no test sweep"}
    json.dump({"summary": summary, "results": results}, open(OUT_JSON, "w"), indent=2)

    print("\n=== HONEST TEST (val/train-frozen threshold, scaler applied) ===")
    print(f"  fused_prob: min={p.min():.4f} max={p.max():.4f} mean={p.mean():.4f} (should overlap val)")
    print(f"  AUC      = {a:.4f}  95% CI [{auc_lo}, {auc_hi}]")
    print(f"  Accuracy = {acc:.4f}  95% CI [{acc_lo}, {acc_hi}]  @thr={thr:.4f}")
    print(f"  (old INVALID test-swept number was 0.8648 @ thr=0.010)")
    for s, d in sorted(per_source.items()):
        print(f"    {s:18s} acc={d['accuracy']:.4f} ({d['correct']}/{d['total']})")
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
