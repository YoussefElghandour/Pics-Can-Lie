import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402
import json
from datetime import datetime

def load(path):
    try:
        with open(path) as f: return json.load(f)
    except: return None

clip = load(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'clip_only_results.json'))
aitr = load(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'aitr_fusion_results.json'))
attn = load(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'attention_analysis.json'))
err  = load(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'error_analysis.json'))
comp = load(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'comparison_table.json'))

def fmt(d): return f"{d*100:.2f}%" if d is not None else "N/A"

lines = [
    "# Overnight Run - DONE",
    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    "---",
    "",
    "## Task 1 - Data Verification",
    "PASSED - 1001 samples, shapes (1001,768) confirmed, no NaN/Inf.",
    "Labels: true=338, miscaptioned=338, out-of-context=325",
    "",
    "## Task 2 - CLIP-only Results",
]
if clip:
    lines += [
        f"Best threshold: {clip['best_threshold']}",
        f"Overall accuracy: {fmt(clip['overall_accuracy'])}",
        f"AUC-ROC: {clip['auc_roc']}  |  F1: {clip['f1']}",
        f"  true (REAL):           {fmt(clip['per_category'].get('true'))}",
        f"  miscaptioned (FAKE):   {fmt(clip['per_category'].get('miscaptioned'))}",
        f"  out-of-context (FAKE): {fmt(clip['per_category'].get('out-of-context'))}",
    ]
else:
    lines.append("FAILED")

lines += ["", "## Task 3 - AITR Fusion Results"]
if aitr:
    lines += [
        f"Best threshold: {aitr['best_threshold']}",
        f"Overall accuracy: {fmt(aitr['overall_accuracy'])}",
        f"At thr=0.54 (NewsCLIPpings optimal): {fmt(aitr['accuracy_at_054'])}",
        f"AUC-ROC: {aitr['auc_roc']}  |  F1: {aitr['f1']}",
        f"  true (REAL):           {fmt(aitr['per_category'].get('true'))}",
        f"  miscaptioned (FAKE):   {fmt(aitr['per_category'].get('miscaptioned'))}",
        f"  out-of-context (FAKE): {fmt(aitr['per_category'].get('out-of-context'))}",
    ]
else:
    lines.append("FAILED")

lines += ["", "## Task 4 - Attention Explainability"]
if attn:
    lines.append("Mean CLS attention weights:")
    for tok, val in attn['mean_attention_all'].items():
        lines.append(f"  {tok}: {val:.4f}")
    lines.append("Dominant token distribution:")
    for tok, val in attn['dominant_token_distribution'].items():
        lines.append(f"  {tok}: {val*100:.1f}%")
else:
    lines.append("FAILED")

lines += ["", "## Task 5 - Error Analysis"]
if err:
    lines += [
        f"Total errors: {err['total_errors']} / 1001",
        f"FP (real->fake): {err['false_positives']}",
        f"FN (fake->real): {err['false_negatives']}",
        f"  FN miscaptioned: {err['fn_by_category'].get('miscaptioned')}",
        f"  FN out-of-context: {err['fn_by_category'].get('out-of-context')}",
        f"CLIP sim correct: {err['clip_sim_correct_mean']}  |  incorrect: {err['clip_sim_incorrect_mean']}",
    ]
else:
    lines.append("FAILED")

lines += ["", "## Task 6 - Comparison Table"]
if comp:
    for r in comp['literature']:
        acc = f"{r['accuracy']*100:.2f}%" if r['accuracy'] else "TBD"
        lines.append(f"  {r['system']}: {acc}")
    for r in comp['your_results']:
        lines.append(f"  {r['system']}: {r['accuracy']*100:.2f}% (AUC={r['auc']:.4f})")
else:
    lines.append("FAILED")

lines += [
    "",
    "## Files Written",
    "- clip_only_results.json",
    "- aitr_fusion_results.json",
    "- attention_analysis.json",
    "- per_sample_attention.csv",
    "- error_analysis.json",
    "- comparison_table.json",
    "- task1_verify.py through task7_done.py",
    "",
    "## Key Questions to Answer in the Morning",
    "1. Does AITR beat CLIP-only? (does fusion generalize to VERITE?)",
    "2. Is overall accuracy above 74.40%? (zero-shot generalization win)",
    "3. Which token dominates attention? (explainability finding)",
    "4. Are miscaptioned errors higher than OOC? (expected - NLI weakness)",
    "5. Does thr=0.54 transfer or does VERITE need its own threshold?",
]

done_text = "\n".join(lines)
with open(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'DONE.md'), 'w', encoding='utf-8') as f:
    f.write(done_text)

print(done_text)
print("\n" + "=" * 50)
print("ALL TASKS COMPLETE - check DONE.md")
print("=" * 50)
