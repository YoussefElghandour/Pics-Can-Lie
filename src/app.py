"""
Pics Can Lie — Multimodal Misinformation Detection demo.

Wired to the ACTUAL thesis system (audit Prompt C):
  * CONSISTENCY (primary): fine-tuned CLIP ViT-L/14 v2 classifier -> AITR fusion,
    decided at the val-frozen threshold with the persisted StandardScaler, so the
    demo's verdict matches the thesis evaluation exactly. This is the only signal
    that survived the audit (test AUC ~0.934).
  * IMAGE ORIGIN (auxiliary, medium): SightEngine live API. NOTE: this is a
    DIFFERENT module than the research one (fine-tuned Ateeq); both have the
    source-distribution caveat from audit Check 3.
  * FACTUALITY (auxiliary, low): DeBERTa-v3 NLI, entailment = softmax[1], premise
    = top-TF-IDF sentences of the provided article (the discriminative v3 method).

Models load lazily on first use so the module can be imported / smoke-tested
without pulling ~2GB of weights.
"""

import os
import json
import re
import tempfile

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths via central config (single PCL_ROOT env var) ───────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import config

CLIP_CKPT     = str(config.CLIP_CKPT)
AITR_CKPT     = str(config.AITR_CKPT)
SCALER_PATH   = str(config.SCALER_PATH)
THRESH_PATH   = str(config.THRESH_PATH)
NLI_MODEL     = "cross-encoder/nli-deberta-v3-large"

# ── SightEngine credentials — env only, NEVER hardcoded (was a leaked secret) ─
SIGHTENGINE_USER   = os.environ.get("SIGHTENGINE_API_USER", "")
SIGHTENGINE_SECRET = os.environ.get("SIGHTENGINE_API_SECRET", "")
SIGHTENGINE_URL    = "https://api.sightengine.com/1.0/check.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Frozen decision threshold on the AITR fused prob (from the test-path fix).
try:
    FROZEN_THRESHOLD = json.load(open(THRESH_PATH))["frozen_threshold"]
except Exception:
    FROZEN_THRESHOLD = 0.55  # fallback if the json is missing


# ── Thesis model definitions ─────────────────────────────────────────────────
class CLIPClassifier(nn.Module):
    """Fine-tuned CLIP v2 head: input [img || txt || cos] (1537) -> 512 -> 128 -> 1."""
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2 + 1, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1))

    def forward(self, images, input_ids):
        img_f = F.normalize(self.clip.encode_image(images), dim=-1)
        txt_f = F.normalize(self.clip.encode_text(input_ids), dim=-1)
        cos = (img_f * txt_f).sum(-1, keepdim=True)
        return torch.sigmoid(self.classifier(torch.cat([img_f, txt_f, cos], -1))).squeeze(-1), img_f, txt_f


class AITR(nn.Module):
    """AITR fusion matching fusion_aitr/aitr_weights.pt (9 scalars)."""
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


# ── Lazy model singletons ────────────────────────────────────────────────────
_M: dict = {}


def _models():
    """Load CLIP v2 + AITR + scaler + DeBERTa on first call; cache thereafter."""
    if _M:
        return _M
    import clip
    import joblib
    print("Loading thesis models (CLIP v2 + AITR + DeBERTa)...")
    clip_base, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    clf = CLIPClassifier(clip_base.float()).to(device)
    clf.load_state_dict(torch.load(CLIP_CKPT, map_location=device)["model_state"])
    clf.eval()
    aitr = AITR().to(device)
    st = torch.load(AITR_CKPT, map_location=device)
    aitr.load_state_dict(st["state_dict"] if "state_dict" in st else st)
    aitr.eval()
    bundle = joblib.load(SCALER_PATH)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device).eval()
    _M.update(clip_preprocess=preprocess, clip_clf=clf, aitr=aitr,
              scaler=bundle["scaler"], train_means=np.array(bundle["train_means"]),
              nli_tok=nli_tok, nli=nli, clip_mod=clip)
    print(f"Models ready on {device}.\n")
    return _M


def extract_top_sentences(article: str, caption: str, top_k: int = 3) -> str:
    """Top-k most caption-relevant article sentences via TF-IDF (NLI premise)."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", article) if len(s.strip()) > 20]
    if len(sentences) <= top_k:
        return article
    tfidf = TfidfVectorizer(stop_words="english").fit_transform([caption] + sentences)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return ". ".join(sentences[i] for i in sorted(top_idx)) + "."


def get_consistency(image: Image.Image, caption: str) -> dict:
    """Primary signal: fine-tuned CLIP v2 -> AITR fused prob at the frozen threshold.

    Returns clip_prob, clip_sim, fused_prob. The 7 evidence/NLI scalars do not
    exist in the live demo, so they are imputed with TRAIN means and standardized
    through the persisted scaler — identical to the thesis test path.
    """
    M = _models()
    img = M["clip_preprocess"](image).unsqueeze(0).to(device)
    tok = M["clip_mod"].tokenize([caption], truncate=True).to(device)
    with torch.no_grad():
        clip_prob, img_f, txt_f = M["clip_clf"](img, tok)
        clip_sim = (img_f * txt_f).sum(-1)
        raw = M["train_means"].copy().reshape(1, -1).astype(np.float32)
        raw[0, 0] = float(clip_prob)   # order: [clip_prob, clip_sim, deberta, s2..s6, wiki]
        raw[0, 1] = float(clip_sim)
        scaled = torch.tensor(M["scaler"].transform(raw), dtype=torch.float32, device=device)
        fused = float(M["aitr"](img_f, txt_f, scaled))
    return {"clip_prob": float(clip_prob), "clip_sim": float(clip_sim), "fused_prob": fused}


def get_nli_entailment_score(article: str, caption: str) -> float:
    """DeBERTa-v3 NLI entailment (softmax index [1]); premise = top sentences."""
    M = _models()
    premise = extract_top_sentences(article, caption)
    inputs = M["nli_tok"](premise, caption, return_tensors="pt",
                          truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        probs = torch.softmax(M["nli"](**inputs).logits, dim=1)[0]
    return float(probs[1])  # entailment index


def get_sightengine_score(image: Image.Image) -> float:
    """Get SightEngine AI-generated detection score."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp, format="JPEG")
        tmp_path = tmp.name

    try:
        params = {
            "models": "genai",
            "api_user": SIGHTENGINE_USER,
            "api_secret": SIGHTENGINE_SECRET,
        }
        with open(tmp_path, "rb") as f:
            files = {"media": ("image.jpg", f)}
            response = requests.post(SIGHTENGINE_URL, files=files, data=params, timeout=30)

        response.raise_for_status()
        data = response.json()

        if data.get("status") != "success":
            return -1.0

        return data.get("type", {}).get("ai_generated", 0.0)
    except Exception:
        return -1.0
    finally:
        os.unlink(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# Decomposed sub-signal thresholds
#
# Each sub-signal has its OWN decision threshold. These are deliberately NOT
# derived from any fused probability — the verdict is anchored to the
# consistency signal only; the other two are reported for transparency.
# ──────────────────────────────────────────────────────────────────────────────

# 1) IMAGE–CAPTION CONSISTENCY  (PRIMARY, high confidence)
#    Source: fine-tuned CLIP v2 classifier -> AITR fused probability.
#    Threshold = the val-frozen threshold from the test-path fix
#    (fusion_aitr/frozen_threshold.json), so the demo verdict matches the thesis
#    evaluation exactly. NOT a hand-picked constant.
CONSISTENCY_THRESHOLD        = FROZEN_THRESHOLD   # fused_prob >= thr → Consistent

# 2) IMAGE ORIGIN  (MEDIUM confidence)
#    Source: SightEngine `genai.ai_generated` probability (0..1, higher = more AI).
#    Original app used 0.5 as the "AI-generated vs Real photo" cut.
IMAGE_ORIGIN_THRESHOLD       = 0.50   # ai_generated >= 0.50 → AI-generated / manipulated

# 3) CAPTION FACTUALITY  (LOW confidence / auxiliary)
#    Source: DeBERTa-v3 NLI entailment probability (article ⊨ caption).
#    Internal ablation shows this signal barely discriminates, so it is marked
#    low-confidence and MUST NOT drive the overall verdict.
FACTUALITY_THRESHOLD         = 0.50   # entailment < 0.50 → Possible distortion


def compute_signals(image: Image.Image, caption: str, article_text: str) -> dict:
    """Run the three modules and return three INDEPENDENT sub-signals plus an
    overall verdict that is driven ONLY by the consistency signal.

    Returns the structured object:
      {
        overall_verdict: "REAL" | "FAKE",
        overall_confidence: float,
        signals: {
          consistency:  { label, score, threshold, confidence_tier: "high" },
          image_origin: { label, score, threshold, confidence_tier: "medium" },
          factuality:   { label, score, threshold, confidence_tier: "low" },
        }
      }
    """
    caption = caption.strip()

    # ── Signal 1: image–caption consistency (CLIP v2 -> AITR, frozen threshold) ─
    cons = get_consistency(image, caption)
    fused = cons["fused_prob"]
    is_consistent = fused >= CONSISTENCY_THRESHOLD
    consistency_label = "Consistent" if is_consistent else "Out of context"
    # Confidence = distance from the frozen threshold, scaled into [0, 1].
    denom = (1.0 - CONSISTENCY_THRESHOLD) if is_consistent else CONSISTENCY_THRESHOLD
    consistency_conf = float(min(max(abs(fused - CONSISTENCY_THRESHOLD) / max(denom, 1e-6), 0.0), 1.0))

    # ── Signal 2: image origin (SightEngine — auxiliary) ──────────────────────
    # NOTE: SightEngine is a DIFFERENT module than the research one (fine-tuned
    # Ateeq). Carries the source-distribution caveat from audit Check 3.
    se_score = get_sightengine_score(image)
    if se_score < 0:
        image_origin_label = "Unavailable (no SightEngine key / API error)"
    elif se_score >= IMAGE_ORIGIN_THRESHOLD:
        image_origin_label = "AI-generated / manipulated [auxiliary]"
    else:
        image_origin_label = "Authentic photo [auxiliary]"

    # ── Signal 3: caption factuality (DeBERTa NLI) ────────────────────────────
    if article_text and article_text.strip():
        nli_score = get_nli_entailment_score(article_text.strip(), caption)
        factuality_label = (
            "Possible distortion" if nli_score < FACTUALITY_THRESHOLD
            else "No distortion detected"
        )
    else:
        nli_score = None
        factuality_label = "Not assessed (no article)"

    # ── Overall verdict: anchored ONLY to the consistency signal ─────────────
    overall_verdict = "REAL" if is_consistent else "FAKE"

    return {
        "overall_verdict":    overall_verdict,
        "overall_confidence": consistency_conf,
        "signals": {
            "consistency": {
                "label":           consistency_label,
                "score":           float(fused),          # AITR fused prob (decision)
                "clip_prob":       cons["clip_prob"],     # underlying CLIP v2 signals
                "clip_sim":        cons["clip_sim"],
                "threshold":       CONSISTENCY_THRESHOLD,
                "confidence_tier": "high",
            },
            "image_origin": {
                "label":           image_origin_label,
                "score":           (float(se_score) if se_score >= 0 else None),
                "threshold":       IMAGE_ORIGIN_THRESHOLD,
                "confidence_tier": "medium",
            },
            "factuality": {
                "label":           factuality_label,
                "score":           (float(nli_score) if nli_score is not None else None),
                "threshold":       FACTUALITY_THRESHOLD,
                "confidence_tier": "low",
            },
        },
    }


# Visual styling per confidence tier.
_TIER_BADGE = {
    "high":   ("#155724", "#d4edda", "HIGH"),
    "medium": ("#856404", "#fff3cd", "MEDIUM"),
    "low":    ("#6c757d", "#e9ecef", "LOW · auxiliary"),
}


def _signal_row(name: str, sig: dict, muted: bool = False) -> str:
    """Render one sub-signal as an HTML row with a confidence-tier badge."""
    fg, bg, badge = _TIER_BADGE[sig["confidence_tier"]]
    score = sig["score"]
    score_txt = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
    opacity = "0.6" if muted else "1.0"
    return (
        f'<div style="display:flex; align-items:center; justify-content:space-between; '
        f'padding:10px 14px; margin:6px 0; border-radius:8px; '
        f'background-color:{bg}; opacity:{opacity};">'
        f'  <div style="flex:2;"><b style="color:{fg};">{name}</b>'
        f'    <div style="font-size:0.85em; color:{fg};">threshold {sig["threshold"]:.2f}</div></div>'
        f'  <div style="flex:2; text-align:center; color:{fg}; font-weight:600;">{sig["label"]}</div>'
        f'  <div style="flex:1; text-align:center; color:{fg};">{score_txt}</div>'
        f'  <div style="flex:1; text-align:right;">'
        f'    <span style="background:{fg}; color:white; border-radius:6px; '
        f'      padding:2px 8px; font-size:0.75em;">{badge}</span></div>'
        f'</div>'
    )


def analyze(image, caption, article_text):
    """Gradio entry point — returns (overall HTML, signals HTML)."""
    if image is None:
        return "Please upload an image.", ""
    if not caption or not caption.strip():
        return "Please enter a caption.", ""

    image = Image.fromarray(image).convert("RGB")
    result = compute_signals(image, caption, article_text or "")

    verdict = result["overall_verdict"]
    conf = result["overall_confidence"]
    sigs = result["signals"]
    is_real = verdict == "REAL"

    # ── Overall verdict banner (anchored to consistency only) ─────────────────
    overall_html = (
        f'<div style="text-align:center; padding:20px; '
        f'background-color:{"#d4edda" if is_real else "#f8d7da"}; '
        f'border-radius:10px; margin-bottom:10px;">'
        f'<h1 style="color:{"#155724" if is_real else "#721c24"}; margin:0;">{verdict}</h1>'
        f'<p style="color:{"#155724" if is_real else "#721c24"}; margin:5px 0 0 0;">'
        f'Overall confidence: {conf:.1%} '
        f'<span style="font-size:0.8em;">(from image–caption consistency)</span></p>'
        f'</div>'
    )

    # ── Three independent sub-signal rows ─────────────────────────────────────
    signals_html = '<h3 style="margin-bottom:4px;">Sub-signals</h3>'
    signals_html += _signal_row("Image–caption consistency", sigs["consistency"])
    signals_html += _signal_row("Image origin", sigs["image_origin"])
    signals_html += _signal_row("Caption factuality", sigs["factuality"], muted=True)
    signals_html += (
        '<p style="font-size:0.8em; color:#6c757d; margin-top:10px;">'
        'Factuality is shown for transparency and is not a reliable fact-checker on its own.'
        '</p>'
    )

    return overall_html, signals_html


# ── Gradio Interface ──
with gr.Blocks(title="Pics Can Lie — Misinformation Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Pics Can Lie — Multimodal Misinformation Detector\n"
        "Upload a news image, enter its caption, and optionally paste the article text. "
        "The system reports an overall verdict plus three **independent** sub-signals."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image", type="numpy")
            caption_input = gr.Textbox(
                label="Caption",
                placeholder="Enter the image caption...",
                lines=2,
            )
            article_input = gr.Textbox(
                label="Article Text (optional — enables factuality signal)",
                placeholder="Paste the article text here...",
                lines=6,
            )
            check_btn = gr.Button("Check", variant="primary", size="lg")

        with gr.Column(scale=1):
            verdict_output = gr.HTML(label="Overall verdict")
            signals_output = gr.HTML(label="Sub-signals")

    check_btn.click(
        fn=analyze,
        inputs=[image_input, caption_input, article_input],
        outputs=[verdict_output, signals_output],
    )

    gr.Markdown(
        "---\n"
        "**Sub-signals:** "
        "Image–caption consistency (BLIP ITM, *high* confidence — drives the verdict) · "
        "Image origin (SightEngine, *medium*) · "
        "Caption factuality (DeBERTa-v3 NLI, *low / auxiliary*)\n\n"
        "*Factuality is shown for transparency and is not a reliable fact-checker on its own.*"
    )

if __name__ == "__main__":
    demo.launch(share=False)
