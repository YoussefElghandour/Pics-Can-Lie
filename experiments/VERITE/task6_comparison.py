import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402
import json

with open(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'clip_only_results.json')) as f:
    clip = json.load(f)
with open(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'aitr_fusion_results.json')) as f:
    aitr = json.load(f)

literature = [
    {'system': 'CLIP ViT-L/14 zero-shot (VERITE paper baseline)', 'accuracy': 0.7440, 'notes': 'No training'},
    {'system': 'VERITE paper best (trained on VERITE)',            'accuracy': 0.8100, 'notes': 'In-domain'},
    {'system': 'LAMAR (new SOTA Apr 2025)',                        'accuracy': None,   'notes': 'Check paper'},
    {'system': 'MAD-Sherlock (ICML 2025)',                         'accuracy': None,   'notes': 'Check paper'},
]
your_results = [
    {'system': 'Your CLIP-only zero-shot',   'accuracy': clip['overall_accuracy'], 'auc': clip['auc_roc'], 'notes': 'Zero-shot'},
    {'system': 'Your AITR fusion zero-shot', 'accuracy': aitr['overall_accuracy'], 'auc': aitr['auc_roc'], 'notes': 'Trained on NewsCLIPpings only'},
]

print("=" * 65)
print("VERITE Comparison Table")
print("=" * 65)
for r in literature:
    acc = f"{r['accuracy']*100:.2f}%" if r['accuracy'] else "TBD"
    print(f"  {r['system']:<52} {acc}")
print("-" * 65)
for r in your_results:
    print(f"  {r['system']:<52} {r['accuracy']*100:.2f}%  AUC={r['auc']:.4f}")
print("=" * 65)

with open(_os.path.join(str(_cfg.ROOT), 'experiments', 'VERITE', 'comparison_table.json'), 'w') as f:
    json.dump({'literature': literature, 'your_results': your_results}, f, indent=2)
print("Task 6 DONE - saved comparison_table.json")
