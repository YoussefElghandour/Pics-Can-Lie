"""
2.2 — fetch_wiki_pages_mmfb.py  (bulk-extracts API edition)
-----------------------------------------------------------
Collect every unique entity from caption_entities_{train,val}.json and
fetch their Wikipedia intro summaries via the bulk extracts endpoint:

    https://en.wikipedia.org/w/api.php
        ?action=query
        &prop=extracts
        &exintro=true
        &explaintext=true
        &titles=T1|T2|...|T50
        &format=json
        &redirects=1

Bulk endpoint quirks we handle:
  1. Response is keyed by page_id, NOT by title. We map back to the
     originally requested titles via the 'normalized' and 'redirects'
     arrays so {requested_title -> page_text} is exact.
  2. Pages with 'missing' key are cached as "" (negative cache) so
     re-runs don't retry them.
  3. Titles containing '|' (rare; spaCy occasionally yields one) are
     sanitized: literal pipes are stripped before batching since pipe
     is the API's title delimiter.
  4. 50 titles per request — the anonymous cap. Don't go higher.
  5. Cache flushed every 10 batches (~500 entities) so a crash near the
     end doesn't lose hours of work.

Resume-safe: re-running picks up wherever cache left off. Negative-cached
misses count as 'done'.

Run:
    python mmfakebench_training/fetch_wiki_pages_mmfb.py
"""


from __future__ import annotations
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402

import json
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wikipedia_factcheck import (  # noqa: E402
    WIKI_API, WIKI_HEADERS, WIKI_MAX_CHARS, WIKI_TIMEOUT,
)

PROJECT_ROOT = Path(str(_cfg.ROOT))
OUT_ROOT     = PROJECT_ROOT / "mmfakebench_training"
CACHE_PATH   = PROJECT_ROOT / "wiki_page_cache.json"

ENT_FILES = [
    OUT_ROOT / "caption_entities_train.json",
    OUT_ROOT / "caption_entities_val.json",
]

BATCH_SIZE          = 50      # anonymous-API cap
INTER_BATCH_DELAY_S = 0.2     # politeness sleep between requests
CACHE_FLUSH_EVERY   = 10      # batches => every ~500 entities
MAX_BATCH_RETRIES   = 3
RETRY_BACKOFF_S     = 2.0


# ── Cache helpers ───────────────────────────────────────────────────────────
def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp.replace(CACHE_PATH)


def cache_keys(entity: str) -> tuple[str, str]:
    raw = entity.strip()
    underscored = raw.replace(" ", "_")
    return raw, underscored


def has_in_cache(cache: dict, entity: str) -> bool:
    raw, und = cache_keys(entity)
    return (raw in cache) or (und in cache)


def store(cache: dict, entity: str, text: str) -> None:
    """Cache under both raw and underscored keys (match existing cache style)."""
    raw, und = cache_keys(entity)
    cache[raw] = text
    cache[und] = text


# ── Title sanitization ──────────────────────────────────────────────────────
_PIPE_RE = re.compile(r"[|]")

def sanitize_title(t: str) -> str:
    """Strip literal pipes (rare spaCy output) and trim whitespace."""
    return _PIPE_RE.sub("", t).strip()


# ── Bulk fetch one batch ────────────────────────────────────────────────────
def fetch_batch(titles: list[str]) -> dict[str, str]:
    """
    Returns {requested_title -> extract_or_empty}. Empty string means
    Wikipedia returned 'missing' or no extract.

    Uses 'normalized' and 'redirects' arrays from the response to map
    canonical page titles back to the titles we actually requested.
    """
    if not titles:
        return {}

    params = {
        "action":      "query",
        "prop":        "extracts",
        "exintro":     True,
        "explaintext": True,
        "titles":      "|".join(titles),
        "redirects":   1,
        "format":      "json",
        "formatversion": 2,
    }

    for attempt in range(1, MAX_BATCH_RETRIES + 1):
        try:
            resp = requests.get(
                WIKI_API, params=params, headers=WIKI_HEADERS, timeout=WIKI_TIMEOUT * 2,
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            if attempt < MAX_BATCH_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)
            else:
                # Hard fail — give every title an empty extract so we don't
                # block forever; resume will retry next run.
                return {t: "" for t in titles}

    query = data.get("query", {}) if isinstance(data, dict) else {}
    pages = query.get("pages", []) or []

    # Build "what title did this page come from" mapping.
    # formatversion=2 returns 'pages' as a LIST with 'title' set to the
    # canonical (post-redirect, post-normalize) title.
    # normalized = [{'from': 'requested', 'to': 'normalized'}]
    # redirects  = [{'from': 'normalized', 'to': 'final_title'}]
    canonical_to_requested: dict[str, list[str]] = {}
    normalized = {n["from"]: n["to"] for n in query.get("normalized", []) or []}
    redirects  = {r["from"]: r["to"] for r in query.get("redirects", []) or []}

    for req in titles:
        cur = req
        # chain through normalization then redirect (one hop each is enough)
        if cur in normalized:
            cur = normalized[cur]
        if cur in redirects:
            cur = redirects[cur]
        canonical_to_requested.setdefault(cur, []).append(req)

    out: dict[str, str] = {t: "" for t in titles}
    for page in pages:
        canonical = page.get("title", "")
        if page.get("missing", False):
            extract = ""
        else:
            extract = (page.get("extract", "") or "").strip()
            extract = extract[:WIKI_MAX_CHARS]
        for req in canonical_to_requested.get(canonical, []):
            out[req] = extract

    # Anything still un-matched (very rare) — leave empty.
    return out


# ── Driver ──────────────────────────────────────────────────────────────────
def main() -> None:
    cache = load_cache()
    print(f"[cache] loaded {len(cache)} entries from {CACHE_PATH.name}")

    # Collect unique entities
    all_entities: set[str] = set()
    for ef in ENT_FILES:
        if not ef.exists():
            sys.exit(f"[!] missing {ef.name} — run 2.1 first")
        with open(ef, encoding="utf-8") as f:
            by_id = json.load(f)
        for ents in by_id.values():
            for e in ents:
                s = sanitize_title(e or "")
                if s:
                    all_entities.add(s)
    print(f"[entities] {len(all_entities)} unique entities across train+val")

    pending = sorted(e for e in all_entities if not has_in_cache(cache, e))
    print(f"[entities] cached={len(all_entities) - len(pending)}, "
          f"to fetch={len(pending)}")

    if not pending:
        print("[done] cache already complete")
        return

    n_hit, n_miss = 0, 0
    n_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE

    with tqdm(total=len(pending), desc="Wikipedia bulk", unit="ent") as pbar:
        for b in range(n_batches):
            batch = pending[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            result = fetch_batch(batch)
            for req, text in result.items():
                store(cache, req, text)
                if text:
                    n_hit += 1
                else:
                    n_miss += 1
            pbar.update(len(batch))

            if (b + 1) % CACHE_FLUSH_EVERY == 0:
                save_cache(cache)
                pbar.set_postfix(hits=n_hit, misses=n_miss, flushed=True)

            time.sleep(INTER_BATCH_DELAY_S)

    save_cache(cache)
    print(f"\n[done] this run: {n_hit} hits, {n_miss} misses")
    print(f"[done] cache now has {len(cache)} entries -> {CACHE_PATH}")


if __name__ == "__main__":
    main()
