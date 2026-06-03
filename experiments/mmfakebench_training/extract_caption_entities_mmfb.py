"""
2.1 — extract_caption_entities_mmfb.py
--------------------------------------
Run spaCy NER (en_core_web_lg) on every MMFakeBench caption (train + val).
Keep entity types: PERSON, ORG, GPE, LOC, EVENT, FAC, NORP.
Reuses the existing helper `extract_entities` from wikipedia_factcheck.py so
the behaviour is identical to the NewsCLIPpings pipeline (deduped, capped at
MAX_ENTITIES=4, with a noun-chunk fallback when NER finds nothing).

Outputs:
  mmfakebench_training/caption_entities_train.json   {sample_id: [entities]}
  mmfakebench_training/caption_entities_val.json     {sample_id: [entities]}

Also prints coverage stats: mean entities per caption,
% zero-entity captions, breakdown by fake_cls.

Run:
    python mmfakebench_training/extract_caption_entities_mmfb.py
"""


from __future__ import annotations
import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402

import json
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# Reuse project pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wikipedia_factcheck import extract_entities, load_spacy  # noqa: E402

MMFB_ROOT = Path(_os.path.join(str(_cfg.MMFB_ROOT)))
OUT_ROOT  = Path(_os.path.join(str(_cfg.ROOT), 'experiments', 'mmfakebench_training'))

SPLITS = {
    "train": MMFB_ROOT / "MMFakeBench_test.json",
    "val":   MMFB_ROOT / "MMFakeBench_val.json",
}


def run_split(name: str, json_path: Path, nlp) -> None:
    out_path = OUT_ROOT / f"caption_entities_{name}.json"
    with open(json_path, encoding="utf-8") as f:
        ann = json.load(f)

    entities_by_id: dict[str, list[str]] = {}
    counts: list[int] = []
    zero_by_cat: Counter[str] = Counter()
    n_by_cat:    Counter[str] = Counter()

    for i, s in enumerate(tqdm(ann, desc=f"[{name}] NER", unit="cap")):
        sid = f"{name}_{i:06d}"
        cap = s.get("text", "") or ""
        ents = extract_entities(cap, nlp)
        entities_by_id[sid] = ents
        counts.append(len(ents))
        cat = s.get("fake_cls", "")
        n_by_cat[cat] += 1
        if len(ents) == 0:
            zero_by_cat[cat] += 1

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entities_by_id, f, ensure_ascii=False)

    # ── Coverage report ──────────────────────────────────────────────────────
    n = len(counts)
    mean_ents = sum(counts) / max(n, 1)
    n_zero = sum(1 for c in counts if c == 0)
    print(f"\n[{name}] saved -> {out_path.name}  ({n} captions)")
    print(f"        mean entities/caption : {mean_ents:.2f}")
    print(f"        zero-entity captions  : {n_zero}/{n} ({n_zero/n:.1%})")
    print(f"        distribution of counts: "
          + ", ".join(f"{k}={v}" for k, v in sorted(Counter(counts).items())))
    print(f"        zero-entity by fake_cls:")
    for cat, total in n_by_cat.most_common():
        z = zero_by_cat.get(cat, 0)
        print(f"          {cat:<35} {z}/{total} ({(z/total if total else 0):.1%})")


def main() -> None:
    nlp = load_spacy()
    for name, jp in SPLITS.items():
        run_split(name, jp, nlp)
    print("\nDone.")


if __name__ == "__main__":
    main()
