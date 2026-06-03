import os as _os, sys as _sys
_h = _os.path.abspath(_os.path.dirname(__file__))
while not _os.path.exists(_os.path.join(_h, 'config.py')) and _os.path.dirname(_h) != _h:
    _h = _os.path.dirname(_h)
_sys.path.insert(0, _h)
import config as _cfg  # noqa: E402
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

model_id = "Ateeqq/ai-vs-human-image-detector"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

processor = AutoImageProcessor.from_pretrained(model_id)
model = SiglipForImageClassification.from_pretrained(model_id).to(device)
model.eval()

labels = model.config.id2label

# Test 1 - real BBC news image
print("\n--- Real Image (BBC) ---")
image = Image.open(_os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'MMFakeBench', 'MMFakeBench_val', 'real', 'bbc_val_50', 'BBC_val_0.png')).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)
for idx, prob in enumerate(probs[0]):
    print(f"{labels[idx]}: {prob.item():.4f}")

# Test 2 - AI-generated DALL-E image
print("\n--- AI-Generated Image (DALL-E) ---")
image = Image.open(_os.path.join(str(_cfg.ROOT), 'datasets', 'dataset', 'MMFakeBench', 'MMFakeBench_val', 'fake', 'fever_AI_val_100', 'fever_dalle_val_1.png')).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)
for idx, prob in enumerate(probs[0]):
    print(f"{labels[idx]}: {prob.item():.4f}")