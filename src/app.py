"""
Pics Can Lie — Multimodal Misinformation Detection demo.

Two-stage pipeline, three independent signals; the verdict is anchored ONLY to
the consistency signal.

  STAGE 1 — CONSISTENCY (primary, drives the verdict)  [NewsCLIPpings track]
    Fine-tuned CLIP ViT-L/14 v2 -> AITR fusion -> ISOTONIC calibration. The raw
    AITR fused prob is over-confident (ECE 0.138); a val-fit isotonic calibrator
    (ECE -> 0.017) maps it to a calibrated P(out-of-context) and the verdict is
    taken at a clean 0.5. Production point: TEST acc 0.865 / AUC 0.932 (blind).
    calibrated_prob = P(out-of-context): >= 0.5 -> out-of-context (FAKE).

  STAGE 2 — wild-image checks (auxiliary)  [MMFakeBench / AI-image track]
    2a. IMAGE ORIGIN (Ateeq, medium): fine-tuned SiglipForImageClassification ->
        P(AI-generated). Validated to generalize to unseen generators (StyleGAN,
        Midjourney) with a known resolution bias on very large real photos.
    2b. FACTUALITY (Claude, low / ON-DEMAND): Claude Sonnet 4.6 + web_search
        verifies the caption's factual claims. Costs money + latency, so it only
        runs when the user clicks the fact-check button. Results are cached.

Models load lazily on first use so the module can be imported / smoke-tested
without pulling the weights. SightEngine and DeBERTa/Wikipedia NLI were removed.
"""

import os
import json
import hashlib

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ── Paths via central config (single PCL_ROOT env var) ───────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import config

CLIP_CKPT     = str(config.CLIP_CKPT)
AITR_CKPT     = str(config.AITR_CKPT)
SCALER_PATH   = str(config.SCALER_PATH)
THRESH_PATH   = str(config.THRESH_PATH)
CALIB_PATH    = str(config.MODELS / "fusion_aitr" / "calibrators.joblib")
ATEEQ_DIR     = str(config.MODELS / "ai_detector_finetuned")

# ── Claude fact-checker (on-demand) ──────────────────────────────────────────
CLAUDE_MODEL    = "claude-sonnet-4-6"
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}
FACTCHECK_CACHE = config.SCRIPTS / "factcheck_llm" / "cache"
FACTCHECK_SYSTEM = (
    "You are a meticulous fact-checking analyst. You are given a single news "
    "caption (a CLAIM). Your job is to decide whether the claim is FACTUALLY "
    "ACCURATE or FACTUALLY DISTORTED.\n\n"
    "Procedure:\n"
    "1. Decompose the caption into its atomic factual claims (who / what / when "
    "/ where / which-event).\n"
    "2. Use the web_search tool to find the underlying real event and "
    "authoritative sources (major outlets, official records).\n"
    "3. Judge whether ALL atomic claims are supported by the evidence. If any "
    "specific fact (a name, date, place, number, event, attribution) is wrong, "
    "the caption is 'distorted'. Identify that specific distorted fact.\n\n"
    "Respond with STRICT JSON only — no prose, no markdown fences. Schema:\n"
    '{ "verdict": "factual" | "distorted", '
    '"confidence": 0.0-1.0, '
    '"distorted_fact": "<the wrong fact, or null>", '
    '"evidence_summary": "<1-2 sentence justification>" }'
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Stage-1 model definitions ────────────────────────────────────────────────
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
    """Load CLIP v2 + AITR + scaler + isotonic + Ateeq on first call; cache thereafter.

    (Claude is NOT loaded here — its client is created on-demand in the fact-check
    path so the demo imports and runs the visual signals without an API key.)
    """
    if _M:
        return _M
    import clip
    import joblib
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    print("Loading models (CLIP v2 + AITR + isotonic + Ateeq)...")

    # Stage 1 — consistency
    clip_base, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    clf = CLIPClassifier(clip_base.float()).to(device)
    clf.load_state_dict(torch.load(CLIP_CKPT, map_location=device)["model_state"])
    clf.eval()
    aitr = AITR().to(device)
    st = torch.load(AITR_CKPT, map_location=device)
    aitr.load_state_dict(st["state_dict"] if "state_dict" in st else st)
    aitr.eval()
    bundle = joblib.load(SCALER_PATH)
    # Val-fit isotonic calibrator: maps the over-confident raw fused prob onto a
    # calibrated P(out-of-context) so the verdict can be taken at a clean 0.5.
    isotonic = joblib.load(CALIB_PATH)["isotonic"]

    # Stage 2a — image origin (Ateeq)
    ateeq_proc = AutoImageProcessor.from_pretrained(ATEEQ_DIR)
    ateeq = AutoModelForImageClassification.from_pretrained(ATEEQ_DIR).to(device).eval()
    ateeq_ai_idx = {v.lower(): int(k) for k, v in ateeq.config.id2label.items()}.get("ai", 0)

    _M.update(clip_preprocess=preprocess, clip_clf=clf, aitr=aitr,
              scaler=bundle["scaler"], train_means=np.array(bundle["train_means"]),
              isotonic=isotonic, clip_mod=clip,
              ateeq_proc=ateeq_proc, ateeq=ateeq, ateeq_ai_idx=ateeq_ai_idx)
    print(f"Models ready on {device}.\n")
    return _M


# ── Stage 1: consistency (CLIP v2 -> AITR -> isotonic) ───────────────────────
def get_consistency(image: Image.Image, caption: str) -> dict:
    """Primary signal: fine-tuned CLIP v2 -> AITR fused prob -> isotonic calibration.

    Returns clip_prob, clip_sim, fused_prob (raw AITR), and calibrated_prob (the
    val-fit isotonic calibration of fused_prob = P(out-of-context); high -> FAKE).
    The 7 evidence/NLI scalars do not exist in the live demo, so they are imputed
    with TRAIN means and standardized through the persisted scaler — identical to
    the thesis test path (and to how the calibrator was fit).
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
    calibrated = float(M["isotonic"].predict(np.array([fused], dtype=np.float64))[0])
    return {"clip_prob": float(clip_prob), "clip_sim": float(clip_sim),
            "fused_prob": fused, "calibrated_prob": calibrated}


# ── Stage 2a: image origin (Ateeq AI-vs-real) ────────────────────────────────
# Ateeq detects whether the IMAGE is AI-generated/manipulated. It is independent
# of the consistency verdict (an authentic photo with a swapped caption is still
# "out of context"). Validated to generalize to unseen generators, with a known
# resolution bias: very large, high-quality real photos can be over-flagged.
LARGE_IMAGE_MP = 12_000_000   # > ~12 MP -> attach the resolution caveat


def get_image_origin(image: Image.Image) -> dict:
    """Ateeq P(AI-generated) for the uploaded image, plus a resolution-bias flag."""
    M = _models()
    px = M["ateeq_proc"](images=image, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        ai_score = float(torch.softmax(M["ateeq"](pixel_values=px).logits, dim=-1)[0, M["ateeq_ai_idx"]])
    w, h = image.size
    return {"ai_score": ai_score, "large": (w * h) > LARGE_IMAGE_MP, "w": w, "h": h}


# ── Stage 2b: factuality (Claude web-search, ON-DEMAND, cached) ───────────────
def get_factuality_claude(caption: str) -> dict:
    """Verify the caption's claims with Claude Sonnet 4.6 + web_search.

    On-demand only (costs money + latency). Cached by caption hash so repeats are
    free. Returns {verdict, confidence, distorted_fact, evidence_summary, status}.
    """
    import re
    caption = caption.strip()
    if not caption:
        return {"status": "error", "evidence_summary": "Empty caption."}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"status": "error",
                "evidence_summary": "ANTHROPIC_API_KEY not set — fact-check unavailable."}

    FACTCHECK_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(caption.encode("utf-8")).hexdigest()[:24]
    cache_path = FACTCHECK_CACHE / f"app_{key}.json"
    if cache_path.exists():
        rec = json.load(open(cache_path, encoding="utf-8"))
        rec["status"] = rec.get("status", "ok") + " (cached)"
        return rec

    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=1024, system=FACTCHECK_SYSTEM,
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": f'Caption to verify:\n"{caption}"'}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        raw = text_blocks[-1].strip() if text_blocks else ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        parsed = json.loads(m.group(0)) if m else None
    except Exception as e:
        return {"status": "error", "evidence_summary": f"{type(e).__name__}: {str(e)[:160]}"}

    if not parsed:
        return {"status": "error", "evidence_summary": "Could not parse Claude response."}
    parsed["status"] = "ok"
    json.dump(parsed, open(cache_path, "w", encoding="utf-8"), indent=2)
    return parsed


# ──────────────────────────────────────────────────────────────────────────────
# Decision thresholds (each sub-signal has its OWN; only consistency drives verdict)
# ──────────────────────────────────────────────────────────────────────────────

# 1) IMAGE–CAPTION CONSISTENCY (PRIMARY) — on the CALIBRATED prob.
#    calibrated_prob = P(out-of-context): >= 0.5 -> out-of-context (FAKE).
CONSISTENCY_THRESHOLD  = 0.50

# 2) IMAGE ORIGIN (Ateeq, MEDIUM) — P(AI-generated) >= 0.50 -> AI-generated.
IMAGE_ORIGIN_THRESHOLD = 0.50


def compute_signals(image: Image.Image, caption: str) -> dict:
    """Run the two fast visual signals (consistency + image origin). Factuality
    (Claude) is intentionally NOT run here — it is on-demand via fact_check()."""
    caption = caption.strip()

    # ── Signal 1: consistency (CLIP v2 -> AITR -> isotonic @0.5) ──────────────
    cons = get_consistency(image, caption)
    calibrated = cons["calibrated_prob"]
    is_out_of_context = calibrated >= CONSISTENCY_THRESHOLD
    is_consistent = not is_out_of_context
    consistency_label = "Out of context" if is_out_of_context else "Consistent"
    consistency_conf = float(calibrated if is_out_of_context else 1.0 - calibrated)

    # ── Signal 2: image origin (Ateeq) ───────────────────────────────────────
    origin = get_image_origin(image)
    ai_score = origin["ai_score"]
    is_ai = ai_score >= IMAGE_ORIGIN_THRESHOLD
    origin_label = "AI-generated / manipulated [auxiliary]" if is_ai else "Authentic photo [auxiliary]"
    if origin["large"]:
        origin_label += " ⚠ very high-res — may over-flag"

    overall_verdict = "REAL" if is_consistent else "FAKE"
    return {
        "overall_verdict":    overall_verdict,
        "overall_confidence": consistency_conf,
        "signals": {
            "consistency": {
                "label":           consistency_label,
                "score":           float(calibrated),        # CALIBRATED P(out-of-context) — decision
                "raw_fused_prob":  float(cons["fused_prob"]),
                "clip_prob":       cons["clip_prob"],
                "clip_sim":        cons["clip_sim"],
                "threshold":       CONSISTENCY_THRESHOLD,
                "confidence_tier": "high",
            },
            "image_origin": {
                "label":           origin_label,
                "score":           float(ai_score),
                "threshold":       IMAGE_ORIGIN_THRESHOLD,
                "confidence_tier": "medium",
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


def analyze(image, caption):
    """Gradio 'Check' handler — runs the fast visual signals (no API)."""
    if image is None:
        return "Please upload an image.", ""
    if not caption or not caption.strip():
        return "Please enter a caption.", ""

    image = Image.fromarray(image).convert("RGB")
    result = compute_signals(image, caption)

    verdict = result["overall_verdict"]
    conf = result["overall_confidence"]
    sigs = result["signals"]
    is_real = verdict == "REAL"

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

    signals_html = '<h3 style="margin-bottom:4px;">Sub-signals</h3>'
    signals_html += _signal_row("Image–caption consistency", sigs["consistency"])
    signals_html += _signal_row("Image origin (AI-generated?)", sigs["image_origin"])
    signals_html += (
        '<p style="font-size:0.8em; color:#6c757d; margin-top:10px;">'
        'Image origin (Ateeq) is independent of the verdict and may over-flag very '
        'large real photos. Use “Fact-check caption” for the factuality signal.'
        '</p>'
    )
    return overall_html, signals_html


def fact_check(caption):
    """Gradio handler — ON-DEMAND Claude web-search factuality (may cost / be slow)."""
    if not caption or not caption.strip():
        return "Please enter a caption first."
    res = get_factuality_claude(caption)
    status = res.get("status", "error")
    if status.startswith("error"):
        return (f'<div style="padding:12px; border-radius:8px; background:#e9ecef; color:#6c757d;">'
                f'Factuality unavailable: {res.get("evidence_summary","")}</div>')
    verdict = res.get("verdict", "?")
    distorted = (verdict == "distorted")
    fg, bg = ("#721c24", "#f8d7da") if distorted else ("#155724", "#d4edda")
    return (
        f'<div style="padding:14px; border-radius:8px; background:{bg}; color:{fg};">'
        f'<b>Caption factuality: {verdict.upper()}</b> '
        f'<span style="font-size:0.8em;">(Claude web search · confidence '
        f'{res.get("confidence","?")}) {("· "+status) if "cached" in status else ""}</span>'
        f'<div style="margin-top:6px; font-size:0.9em;">{res.get("evidence_summary","")}</div>'
        + (f'<div style="margin-top:4px; font-size:0.85em;"><b>Distorted fact:</b> '
           f'{res.get("distorted_fact")}</div>' if res.get("distorted_fact") else "")
        + '<div style="margin-top:6px; font-size:0.75em; opacity:0.8;">'
          'Uses live external web search; not a substitute for human verification.</div>'
        f'</div>'
    )


# ── Gradio Interface ──
with gr.Blocks(title="Pics Can Lie — Misinformation Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Pics Can Lie — Multimodal Misinformation Detector\n"
        "Upload a news image and its caption. The verdict (REAL / FAKE) comes from "
        "**image–caption consistency**; two auxiliary signals — **image origin** "
        "(is the image AI-generated?) and **caption factuality** (Claude web search, "
        "on-demand) — are reported independently."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image", type="numpy")
            caption_input = gr.Textbox(
                label="Caption",
                placeholder="Enter the image caption...",
                lines=2,
            )
            check_btn = gr.Button("Check", variant="primary", size="lg")
            factcheck_btn = gr.Button("Fact-check caption (Claude · web search, may cost)", size="sm")

        with gr.Column(scale=1):
            verdict_output = gr.HTML(label="Overall verdict")
            signals_output = gr.HTML(label="Sub-signals")
            factuality_output = gr.HTML(label="Caption factuality")

    check_btn.click(fn=analyze, inputs=[image_input, caption_input],
                    outputs=[verdict_output, signals_output])
    factcheck_btn.click(fn=fact_check, inputs=[caption_input], outputs=[factuality_output])

    gr.Markdown(
        "---\n"
        "**Sub-signals:** "
        "Image–caption consistency (CLIP → AITR → isotonic, *high* — drives the verdict) · "
        "Image origin (Ateeq AI-detector, *medium*) · "
        "Caption factuality (Claude Sonnet + web search, *low / on-demand*)\n\n"
        "*Auxiliary signals do not override the consistency verdict. Factuality uses "
        "live external web search.*"
    )

if __name__ == "__main__":
    demo.launch(share=False)
