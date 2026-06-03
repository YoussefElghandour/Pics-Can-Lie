"""
DEPRECATED — DO NOT USE (audit Prompt A / Part-4 cleanup).

This script has TWO known bugs and is superseded by run_test_inference_fixed.py:
  * WRONG SCALAR ORDER: it assembles [deberta, s2..s6, wiki, clip_sim, clip_prob]
    (see SCALAR_MEANS / base_scalars below), whereas training used
    [clip_prob, clip_sim, deberta, s2..s6, wiki]. The notebook that produced the
    saved results used the correct order; this .py never matched it.
  * NO StandardScaler: feeds raw scalars, so AITR sees out-of-distribution inputs.
It also used hardcoded source thresholds. Use run_test_inference_fixed.py, which
loads fusion_aitr/scalar_scaler.joblib + fusion_aitr/frozen_threshold.json and
reports AUC + a held-out-frozen-threshold accuracy with bootstrap CIs.

Original docstring:
  End-to-end inference on the NewsCLIPpings test set:
    1. CLIP ViT-L/14 + fine-tuned classifier head (clip_finetuned_v2/clip_classifier.pt)
    2. AITR fusion model (fusion_aitr/aitr_weights.pt) with 9-d scalar features.
    3. Source-aware thresholding for the final decision.
"""

import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip

# ──────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = r"D:\Pics Can Lie"
TEST_ANN      = rf"{PROJECT_ROOT}\dataset\data\NewsClipPings\merged_balanced\test.json"
TEST_META     = rf"{PROJECT_ROOT}\dataset\data\NewsClipPings\metadata\test.json"
IMAGES_ROOT   = rf"{PROJECT_ROOT}\dataset\origin\origin"
CLIP_CKPT     = rf"{PROJECT_ROOT}\clip_finetuned_v2\clip_classifier.pt"
AITR_CKPT     = rf"{PROJECT_ROOT}\fusion_aitr\aitr_weights.pt"
OUT_DIR       = rf"{PROJECT_ROOT}\test_set_results"
OUT_JSON      = os.path.join(OUT_DIR, "test_results.json")

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Means for missing evidence / NLI scalars (from val set).
SCALAR_MEANS = {
    "deberta": 0.05,
    "s2":      0.731,
    "s3":      0.235,
    "s4":      0.620,
    "s5":      0.188,
    "s6":      0.573,
    "wiki":    0.05,
}

# Source-aware decision thresholds on the fused probability.
SOURCE_THRESHOLDS = {
    "bbc":              0.58,
    "washington_post":  0.57,
    "guardian":         0.54,
    "usa_today":        0.44,
}
DEFAULT_THRESHOLD = 0.50

os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
class CLIPClassifier(nn.Module):
    """Matches checkpoint: input = [img || txt || cosine_sim] (1537),
       hidden 512 → 128 → 1, attribute name `classifier`."""
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        dim = 768
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2 + 1, 512),   # 0
            nn.BatchNorm1d(512),           # 1
            nn.ReLU(),                     # 2
            nn.Dropout(0.5),               # 3
            nn.Linear(512, 128),           # 4
            nn.BatchNorm1d(128),           # 5
            nn.ReLU(),                     # 6
            nn.Dropout(0.5),               # 7
            nn.Linear(128, 1),             # 8
        )

    def forward(self, images, input_ids):
        img_f = self.clip.encode_image(images)
        txt_f = self.clip.encode_text(input_ids)
        img_f = F.normalize(img_f, dim=-1)
        txt_f = F.normalize(txt_f, dim=-1)
        cos   = (img_f * txt_f).sum(dim=-1, keepdim=True)
        x     = torch.cat([img_f, txt_f, cos], dim=-1)
        return torch.sigmoid(self.classifier(x)).squeeze(-1), img_f, txt_f


class AITR(nn.Module):
    """Matches fusion_aitr/aitr_weights.pt:
         scalar_proj  = Sequential(Linear(9,768), LayerNorm(768))
         type_embedding(5,768), cls_token(1,1,768)
         transformer = TransformerEncoder(8 heads, 2 layers, d_model=768)
         classifier  = Sequential(LayerNorm(768), Linear(768,256), GELU,
                                  Dropout, Linear(256,1))"""
    def __init__(self, embed_dim=768, scalar_dim=9, num_heads=8,
                 num_layers=2, dropout=0.3, hidden_dim=256):
        super().__init__()
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.type_embedding = nn.Embedding(5, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.classifier  = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, img_emb, txt_emb, scalar):
        B = img_emb.size(0)
        scalar_emb = self.scalar_proj(scalar)
        tokens   = torch.stack(
            [img_emb, txt_emb, img_emb * txt_emb, img_emb - txt_emb, scalar_emb],
            dim=1,
        )
        type_ids = torch.arange(5, device=tokens.device).unsqueeze(0).expand(B, -1)
        tokens   = tokens + self.type_embedding(type_ids)
        cls      = self.cls_token.expand(B, -1, -1)
        out      = self.transformer(torch.cat([cls, tokens], dim=1))
        return torch.sigmoid(self.classifier(out[:, 0])).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
def resolve_image_path(rel_path: str) -> str:
    """metadata stores e.g. 'visual_news/origin/bbc/123/foo.jpg' — strip the prefix."""
    rel = rel_path.replace("\\", "/")
    for prefix in ("visual_news/origin/", "origin/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break
    return os.path.join(IMAGES_ROOT, rel.replace("/", os.sep))


class TestDataset(Dataset):
    def __init__(self, samples, preprocess):
        self.samples    = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = self.preprocess(Image.open(s["image_path"]).convert("RGB"))
        tok = clip.tokenize([s["caption"]], truncate=True)[0]
        return img, tok, idx


def build_samples():
    with open(TEST_ANN, "r", encoding="utf-8") as f:
        ann_data = json.load(f)
    with open(TEST_META, "r", encoding="utf-8") as f:
        meta = json.load(f)

    samples, skipped = [], 0
    for a in ann_data["annotations"]:
        # Caption comes from the article (id); image comes from the
        # displayed image (image_id, which differs for falsified pairs).
        cap_key = str(a["id"])
        img_key = str(a["image_id"])
        if cap_key not in meta or img_key not in meta:
            skipped += 1
            continue
        cap_entry = meta[cap_key]
        img_entry = meta[img_key]
        img_path  = resolve_image_path(img_entry["image_path"])
        if not os.path.exists(img_path):
            skipped += 1
            continue
        samples.append({
            "id":         a["id"],
            "image_id":   a["image_id"],
            "falsified":  bool(a["falsified"]),
            "source":     (a.get("source_dataset")
                           or cap_entry.get("source")
                           or "unknown"),
            "caption":    cap_entry["caption"],
            "image_path": img_path,
        })
    print(f"Loaded {len(samples)} samples ({skipped} skipped).")
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    samples = build_samples()

    # ── CLIP + fine-tuned head ────────────────────────────────────────────
    print("Loading CLIP ViT-L/14 ...")
    clip_base, clip_preprocess = clip.load("ViT-L/14", device=DEVICE, jit=False)
    clip_base = clip_base.float()

    clf  = CLIPClassifier(clip_base).to(DEVICE)
    ckpt = torch.load(CLIP_CKPT, map_location=DEVICE)
    clf.load_state_dict(ckpt["model_state"])
    clf.eval()
    print(f"CLIP classifier loaded (epoch {ckpt.get('epoch', '?')}).")

    # ── AITR fusion ───────────────────────────────────────────────────────
    print("Loading AITR fusion ...")
    aitr = AITR().to(DEVICE)
    aitr_state = torch.load(AITR_CKPT, map_location=DEVICE)
    if isinstance(aitr_state, dict) and "state_dict" in aitr_state:
        aitr_state = aitr_state["state_dict"]
    aitr.load_state_dict(aitr_state)
    aitr.eval()

    # ── Build scalar template (deberta, s2..s6, wiki, clip_sim, clip_prob)
    # Order matches the 9-d vector used at training time:
    # [deberta, s2, s3, s4, s5, s6, wiki, clip_sim, clip_prob].
    base_scalars = torch.tensor([
        SCALAR_MEANS["deberta"],
        SCALAR_MEANS["s2"], SCALAR_MEANS["s3"], SCALAR_MEANS["s4"],
        SCALAR_MEANS["s5"], SCALAR_MEANS["s6"],
        SCALAR_MEANS["wiki"],
        0.0,   # clip cosine sim (filled per-sample)
        0.0,   # clip prob       (filled per-sample)
    ], dtype=torch.float32)

    # ── Inference loop ────────────────────────────────────────────────────
    dataset = TestDataset(samples, clip_preprocess)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    results = []
    with torch.no_grad():
        for imgs, toks, idxs in loader:
            imgs = imgs.to(DEVICE, non_blocking=False)
            toks = toks.to(DEVICE, non_blocking=False)

            clip_prob, img_f, txt_f = clf(imgs, toks)
            clip_sim = (img_f * txt_f).sum(dim=-1)

            B = imgs.size(0)
            scalars = base_scalars.unsqueeze(0).expand(B, -1).clone().to(DEVICE)
            scalars[:, 7] = clip_sim
            scalars[:, 8] = clip_prob

            fused_prob = aitr(img_f, txt_f, scalars)

            for i in range(B):
                s = samples[int(idxs[i])]
                thr  = SOURCE_THRESHOLDS.get(s["source"].lower(), DEFAULT_THRESHOLD)
                p    = float(fused_prob[i].item())
                pred = bool(p >= thr)
                results.append({
                    "id":           s["id"],
                    "image_id":     s["image_id"],
                    "source":       s["source"],
                    "label":        s["falsified"],
                    "clip_prob":    float(clip_prob[i].item()),
                    "clip_sim":     float(clip_sim[i].item()),
                    "fused_prob":   p,
                    "threshold":    thr,
                    "prediction":   pred,
                    "correct":      pred == s["falsified"],
                })

    # ── Metrics ───────────────────────────────────────────────────────────
    n_total   = len(results)
    n_correct = sum(r["correct"] for r in results)
    overall   = n_correct / n_total if n_total else 0.0

    per_source = {}
    for r in results:
        d = per_source.setdefault(r["source"], {"total": 0, "correct": 0})
        d["total"]   += 1
        d["correct"] += int(r["correct"])
    for src, d in per_source.items():
        d["accuracy"] = d["correct"] / d["total"] if d["total"] else 0.0

    summary = {
        "n":               n_total,
        "overall_accuracy": overall,
        "per_source":      per_source,
        "thresholds":      {**SOURCE_THRESHOLDS, "default": DEFAULT_THRESHOLD},
        "scalar_means":    SCALAR_MEANS,
    }

    print("\n=== Results ===")
    print(f"Overall accuracy: {overall:.4f}  ({n_correct}/{n_total})")
    for src, d in sorted(per_source.items()):
        print(f"  {src:20s} {d['accuracy']:.4f}  ({d['correct']}/{d['total']})")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
