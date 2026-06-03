# Stage 1 Colab Handoff — Wikipedia Fact-Check (v2 RAG)

This bundle resumes a partially-completed evaluation of MMFakeBench (1000 captions)
using a retrieval-augmented Wikipedia + DeBERTa-NLI pipeline.

A clean 215-row checkpoint is included. The run will resume from row 215 automatically
and process the remaining ~785 samples. Expected runtime: **~25 minutes** on a Colab
T4/A100 GPU with a stable connection.

---

## 1. Required Python Version

Python **3.10 or higher**. Colab's default runtime (Python 3.10+) is compatible.

---

## 2. pip Installs

Run once at the top of your notebook (or see Cell 1 in `run_stage1_colab.ipynb`):

```bash
pip install -q \
    torch torchvision \
    transformers>=4.38 \
    accelerate \
    sentence-transformers \
    spacy \
    scikit-learn \
    pandas \
    requests \
    numpy
```

> `sentence-transformers` is included for completeness; the current pipeline uses
> `scikit-learn` TF-IDF for reranking. `accelerate` is required by HuggingFace
> for efficient model loading.

---

## 3. spaCy Model Download

After installing spaCy, download the large English model:

```bash
python -m spacy download en_core_web_lg
```

This is ~560 MB and only needs to run once per Colab session.

---

## 4. Directory Structure in Colab

Place all files in `/content/` (Colab's working directory). The expected layout is:

```
/content/
├── wikipedia_factcheck.py               # v1 shared utilities
├── wikipedia_factcheck_v2.py            # v2 RAG pipeline + circuit breaker
├── path_overrides.py                    # patches Windows paths → /content/
├── run_colab.py                         # Colab-aware entry point
├── mmfakebench_factcheck_checkpoint_v2.csv   # 215-row clean checkpoint
├── mmfakebench_factcheck_scores.csv          # v1 baseline (1000 rows, for comparison)
└── dataset/
    └── MMFakeBench/
        └── MMFakeBench_val.json              # 1000-caption evaluation set
```

Upload all files from this bundle using the file upload cell in the notebook
(or `from google.colab import files; files.upload()`).

**Important:** The `dataset/MMFakeBench/` subdirectory must be created before
uploading `MMFakeBench_val.json`:

```python
import os
os.makedirs("/content/dataset/MMFakeBench", exist_ok=True)
```

---

## 5. Why `run_colab.py` Instead of `wikipedia_factcheck_v2.py`

`wikipedia_factcheck.py` hardcodes:
```python
PROJECT_ROOT = Path(r"D:\Pics Can Lie")
```

This path doesn't exist on a Linux Colab instance. `path_overrides.py` patches all
affected module-level constants to `/content/` before any evaluation code runs.
`run_colab.py` is a thin wrapper that imports `path_overrides` first, then calls
the exact same evaluation and summary functions.

**Do not run `wikipedia_factcheck_v2.py` directly in Colab** — it will fail with
`FileNotFoundError` on the dataset path.

---

## 6. Verify Connectivity Before Launching

```python
from path_overrides import *          # applies path patches
from wikipedia_factcheck_v2 import check_wikipedia_connectivity
print("Wikipedia connectivity:", check_wikipedia_connectivity())
```

Only proceed to the main run if this returns `True`.

---

## 7. Resume from Checkpoint (Main Run)

```bash
python run_colab.py
```

The checkpoint file (`mmfakebench_factcheck_checkpoint_v2.csv`, 215 rows) is
automatically detected. The run will skip rows 0–214 and begin at row 215.

If you want a quick smoke-test first (e.g., 10 samples):

```bash
python run_colab.py --max-samples 10
```

---

## 8. Circuit Breaker Behaviour

The script includes a circuit breaker that trips after **15 consecutive
`no_wiki_result`** outcomes (indicating a WiFi/network drop):

- It waits up to **5 minutes**, retrying connectivity every 30 seconds.
- If connectivity is restored: logs `Connectivity restored` and continues.
- If still down after 5 minutes: saves the checkpoint and exits cleanly.

If the circuit breaker trips, simply re-run `python run_colab.py` — it will
resume from the latest checkpoint with no data loss.

A **heartbeat check** also fires every 100 samples as a proactive probe.

---

## 9. Expected Runtime

| Segment | Rows | Estimated time |
|---------|------|----------------|
| Already done (checkpoint) | 0–214 | — |
| Remaining | 215–999 | ~22–28 minutes |
| **Total** | **1000** | **~25 minutes** |

Times assume a Colab T4 GPU and stable Wikipedia connectivity. The NLI model
(DeBERTa-v3-large) runs on GPU; Wikipedia fetches are the main bottleneck.

---

## 10. Output Files — What to Download

After the run completes, download these two files:

| File | Description |
|------|-------------|
| `/content/mmfakebench_factcheck_scores_v2.csv` | **Primary output** — 1000-row final scores |
| `/content/mmfakebench_factcheck_checkpoint_v2.csv` | Checkpoint (identical content, kept for safety) |

```python
from google.colab import files
files.download("/content/mmfakebench_factcheck_scores_v2.csv")
files.download("/content/mmfakebench_factcheck_checkpoint_v2.csv")
```

---

## 11. Comparison Summary (V1 vs V2)

After the main run completes, generate the full comparison table:

```bash
python run_colab.py --summary-only
```

This requires both `/content/mmfakebench_factcheck_scores.csv` (v1 baseline,
included in this bundle) and the freshly-written v2 scores CSV.

---

## Known Windows-to-Linux Path Issues

| Issue | Location | Handled by |
|-------|----------|------------|
| `PROJECT_ROOT = Path(r"D:\Pics Can Lie")` | `wikipedia_factcheck.py:40` | `path_overrides.py` |
| `VAL_SEARCH_PATHS` list (computed at import) | `wikipedia_factcheck.py:45–51` | `path_overrides.py` (patches list directly) |
| `OUT_CSV_V2`, `CKPT_CSV_V2`, `OUT_CSV_V1` | `wikipedia_factcheck_v2.py:67–69` | `path_overrides.py` + explicit args in `run_colab.py` |
| `OUT_CSV`, `CKPT_CSV` | `wikipedia_factcheck.py:41–42` | `path_overrides.py` |

No manual edits to the source files are required.
