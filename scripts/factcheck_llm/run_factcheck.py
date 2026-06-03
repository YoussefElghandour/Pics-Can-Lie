"""
run_factcheck.py — Retrieval-augmented fact-checker for the MMFakeBench
textual-distortion problem, using the Anthropic Claude API + web_search tool.

Self-contained: does NOT touch the AITR pipeline. Everything lives under
  D:\\Pics Can Lie\\factcheck_llm\\

Per sample, the caption (the claim) is decomposed into atomic facts and verified
against real web evidence; the model returns a STRICT-JSON verdict
(factual | distorted). Each result is cached so re-runs are free / resumable.

Usage:
    set ANTHROPIC_API_KEY=sk-...            (never hardcoded)
    python factcheck_llm/run_factcheck.py --limit 5     # smoke test first
    python factcheck_llm/run_factcheck.py               # full 600

The 600-sample eval set is balanced:
    original                       (300, gt_answers='True')  -> expected "factual"
    textual_veracity_distortion    (300, gt_answers='Fake')  -> expected "distorted"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Load ANTHROPIC_API_KEY from the repo-root .env (gitignored).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except Exception:
    pass

# ── Config (edit MODEL here if you want a different one) ──────────────────────
MODEL       = "claude-sonnet-4-6"
MAX_TOKENS  = 1024
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

THIS_DIR   = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent))  # repo root (scripts/factcheck_llm -> root)
import config
CACHE_DIR  = THIS_DIR / "cache"
RESULTS    = THIS_DIR / "results.json"
VAL_JSON   = config.MMFB_ROOT / "MMFakeBench_val.json"

# category -> (expected verdict, ground-truth gt_answers value)
EVAL_CATEGORIES = {
    "original":                    ("factual",   "True"),
    "textual_veracity_distortion": ("distorted", "Fake"),
}

# Rate limiting
BASE_SLEEP   = 1.0     # polite delay between calls (seconds)
MAX_RETRIES  = 6       # for 429 / 529 backoff
BACKOFF_BASE = 2.0

SYSTEM_PROMPT = (
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


def sample_id_from_path(image_path: str) -> str:
    """Stable, filesystem-safe id derived from the record's image_path."""
    stem = image_path.lstrip("/").replace("/", "_").replace("\\", "_")
    stem = re.sub(r"\.(png|jpg|jpeg|webp)$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stem)


def load_eval_set() -> list[dict]:
    """Load the balanced 600-sample eval set from MMFakeBench_val.json."""
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for rec in data:
        cat = rec.get("fake_cls")
        if cat not in EVAL_CATEGORIES:
            continue
        expected, gt_value = EVAL_CATEGORIES[cat]
        # Sanity: the label convention uses 'Fake' (NOT 'False').
        is_fake = rec.get("gt_answers") == "Fake"
        samples.append({
            "sample_id": sample_id_from_path(rec["image_path"]),
            "image_path": rec["image_path"],
            "caption":   rec["text"],
            "fake_cls":  cat,
            "gt_answers": rec.get("gt_answers"),
            "true_label": "distorted" if is_fake else "factual",
            "expected":  expected,
        })
    # Deterministic ordering so --limit always grabs the same prefix.
    samples.sort(key=lambda s: s["sample_id"])
    return samples


def extract_json(message) -> dict | None:
    """Extract the final text block's JSON from a (possibly multi-block) response.

    The response may interleave text + server_tool_use (web_search) +
    web_search_tool_result blocks; the model's final answer is the LAST text
    block. We parse JSON from that, tolerating accidental ```json fences.
    """
    text_blocks = [b.text for b in message.content if getattr(b, "type", None) == "text"]
    if not text_blocks:
        return None
    raw = text_blocks[-1].strip()
    # Strip markdown fences if the model added them despite instructions.
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Grab the outermost JSON object.
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call_claude(client, caption: str):
    """One API call with web search enabled, with 429/529 exponential backoff."""
    import anthropic

    for attempt in range(MAX_RETRIES):
        try:
            return client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=[WEB_SEARCH_TOOL],
                messages=[{"role": "user", "content": f'Caption to verify:\n"{caption}"'}],
            )
        except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
            wait = BACKOFF_BASE ** attempt
            print(f"    [backoff] {type(e).__name__}; sleeping {wait:.1f}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529):
                wait = BACKOFF_BASE ** attempt
                print(f"    [backoff] status {e.status_code}; sleeping {wait:.1f}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exceeded max retries on rate-limit/overload.")


def score_sample(client, sample: dict) -> dict:
    """Call Claude once, parse JSON, retry once on parse failure, else 'error'."""
    msg = call_claude(client, sample["caption"])
    parsed = extract_json(msg)

    if parsed is None:
        # One retry, then log as error rather than crashing the run.
        msg = call_claude(client, sample["caption"])
        parsed = extract_json(msg)

    usage = getattr(msg, "usage", None)
    record = {
        **sample,
        "model": MODEL,
        "verdict":          (parsed or {}).get("verdict"),
        "confidence":       (parsed or {}).get("confidence"),
        "distorted_fact":   (parsed or {}).get("distorted_fact"),
        "evidence_summary": (parsed or {}).get("evidence_summary"),
        "status":           "ok" if parsed else "error",
        "usage": {
            "input_tokens":  getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        } if usage else None,
    }
    if parsed is None:
        record["raw_text"] = "".join(
            getattr(b, "text", "") for b in msg.content
            if getattr(b, "type", None) == "text"
        )[:2000]
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="Claude web-search fact-checker (MMFakeBench).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N samples (cost guard / smoke test).")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY is not set. `set ANTHROPIC_API_KEY=sk-...`")

    import anthropic
    client = anthropic.Anthropic()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_eval_set()
    if args.limit is not None:
        samples = samples[: args.limit]

    n_cat = {}
    for s in samples:
        n_cat[s["fake_cls"]] = n_cat.get(s["fake_cls"], 0) + 1
    print(f"Model: {MODEL}   eval samples: {len(samples)}   by category: {n_cat}")
    print(f"Cache: {CACHE_DIR}\n")

    n_done = n_new = n_err = 0
    for i, sample in enumerate(samples, 1):
        cache_path = CACHE_DIR / f"{sample['sample_id']}.json"
        if cache_path.exists():
            n_done += 1
            continue

        try:
            record = score_sample(client, sample)
        except Exception as e:  # never crash the whole run on one sample
            record = {**sample, "status": "error", "error": f"{type(e).__name__}: {e}"}

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        n_new += 1
        n_err += int(record["status"] == "error")
        flag = "ERR " if record["status"] == "error" else ""
        print(f"[{i}/{len(samples)}] {flag}{sample['fake_cls']:<28} "
              f"verdict={record.get('verdict')}  (expected {sample['expected']})")

        time.sleep(BASE_SLEEP)

    print(f"\nDone. cached_already={n_done}  newly_scored={n_new}  errors={n_err}")
    print(f"Next: python factcheck_llm/evaluate.py")


if __name__ == "__main__":
    main()
