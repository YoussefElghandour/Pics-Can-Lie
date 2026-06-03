"""
deberta_rescore_v3.py  (Prompt B, Part 3)

Recompute the full 5000-row NewsCLIPpings DeBERTa NLI scalar with the SAME
correct method as the discriminative notebook (text_nli_deberta.ipynb), fixing
the non-discriminative deberta_val_scores_v2.csv (REAL 0.415 vs FAKE 0.422).

Root cause of the v2 bug: v2 used the CAPTION's own article as the premise, so
every caption trivially entailed its own article (REAL≈FAKE). The correct method
uses the IMAGE's original article (image_id) as the premise — for falsified pairs
the image's article does NOT support the caption, so entailment drops.

Method (identical to the discriminative notebook):
  premise    = top-3 TF-IDF sentences of the IMAGE's article (by image_id),
  hypothesis = the caption (from annotation id),
  entailment = softmax(logits)[entailment_idx], with entailment_idx auto-detected
               from model.config.id2label and a self-entailment sanity check that
               ABORTS if a known-entailing pair scores < 0.85.

Output: deberta_val_scores_v3.csv  (columns: id, entailment_score, label)

This is a GPU re-scoring pass over 5000 article/caption pairs — run on the 4060:
    python deberta_rescore_v3.py
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import config

VAL_ANN  = str(config.VAL_ANN)
VAL_META = str(config.VAL_META)
ARTICLE_BASE = str(config.ARTICLE_BASE)
VAL_IDS = str(config.CLIP_VAL_FEATURES / "val_sample_ids.csv")
OUT_CSV = str(config.DEBERTA_V3_CSV)
MODEL   = "cross-encoder/nli-deberta-v3-large"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_article_path(meta_article_path: str) -> str:
    rel = meta_article_path.replace("visual_news/", "", 1)
    return os.path.join(ARTICLE_BASE, rel)


def load_article_text(path: str, max_chars: int = 5000) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(max_chars)


def extract_top_sentences(article: str, caption: str, top_k: int = 3) -> str:
    sents = [s.strip() for s in re.split(r"[.!?]+", article) if len(s.strip()) > 20]
    if len(sents) <= top_k:
        return article
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform([caption] + sents)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
    top = sorted(sims.argsort()[-top_k:][::-1])
    return ". ".join(sents[i] for i in top) + "."


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(DEVICE).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    ent_idx = {v.lower(): k for k, v in id2label.items()}["entailment"]
    print(f"id2label={id2label}  entailment_idx={ent_idx}")

    # Self-entailment sanity check — abort if the index resolved wrong.
    with torch.no_grad():
        p = F.softmax(model(**tok("The sky is blue.", "The sky is blue.",
                                  return_tensors="pt").to(DEVICE)).logits, dim=-1)[0]
    print(f"[sanity] self-entailment probs={[round(float(x),4) for x in p]}  ent={float(p[ent_idx]):.4f}")
    assert float(p[ent_idx]) >= 0.85, "Entailment index resolved wrong (self-entailment < 0.85)."

    ann = json.load(open(VAL_ANN, encoding="utf-8"))["annotations"]
    meta = json.load(open(VAL_META, encoding="utf-8"))
    # Key by (id, falsified): each article id has a REAL and a FALSIFIED
    # annotation. A plain {id: a} dict overwrites the real with the falsified, so
    # every premise would use the MISMATCHED image's article -> no separation.
    # The val row's label selects the correct annotation (and thus image_id).
    by_key = {(str(a["id"]), bool(a["falsified"])): a for a in ann}

    want = pd.read_csv(VAL_IDS); want["id"] = want["id"].astype(str)
    rows = []
    for _, w in want.iterrows():
        aid = w["id"]; a = by_key.get((aid, bool(w["label"])))
        if a is None:
            rows.append({"id": aid, "entailment_score": np.nan, "label": int(w["label"])}); continue
        img_key = str(a["image_id"])           # premise = the IMAGE's article (the fix)
        if img_key not in meta:
            rows.append({"id": aid, "entailment_score": np.nan, "label": int(w["label"])}); continue
        cap = meta[str(a["id"])]["caption"]
        art_path = resolve_article_path(meta[img_key].get("article_path", ""))
        try:
            article = load_article_text(art_path)
            premise = extract_top_sentences(article, cap)
        except Exception:
            rows.append({"id": aid, "entailment_score": np.nan, "label": int(w["label"])}); continue
        with torch.no_grad():
            inp = tok(premise, cap, return_tensors="pt", truncation=True,
                      max_length=512, padding=True).to(DEVICE)
            ent = float(F.softmax(model(**inp).logits, dim=-1)[0][ent_idx])
        rows.append({"id": aid, "entailment_score": ent, "label": int(w["label"])})
        if len(rows) % 500 == 0:
            print(f"  {len(rows)}/{len(want)} scored")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    valid = df.dropna(subset=["entailment_score"])
    rm = valid[valid.label == 0]["entailment_score"].mean()
    fm = valid[valid.label == 1]["entailment_score"].mean()
    print(f"\nSaved {OUT_CSV}  ({len(valid)}/{len(df)} scored)")
    print(f"REAL mean={rm:.4f}  FAKE mean={fm:.4f}  gap={rm-fm:+.4f}")
    print("Expect REAL >> FAKE if the fix worked (v2 was 0.415 vs 0.422 — non-discriminative).")
    print("\nRe-test: set deberta source to this CSV in fix_newsclip_inference.build_raw_scalars")
    print("and re-run to see whether corrected NLI changes val AUC vs the broken version.")


if __name__ == "__main__":
    main()
