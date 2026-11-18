---
license: mit
tags:
- intent-classification
- lead-mining
- distillation
- embeddinggemma
- cpu
metrics:
- precision
- accuracy
---

# Tweet Buy-Intent Head (embeddinggemma-2 + MLP)

A 199 KB classification head distilled from a Gemini 2.5 Flash teacher:
scores X/Twitter posts 0-1 for **buying intent** ("is there an app that…",
"anyone recommend…", "I wish there was…").

Base embedder (frozen): **google/embeddinggemma-2** (768-d, Apache-2.0).
The head itself is an MLP 768→64→1 (49k params) — runs on CPU.

## Results (held-out test, n=1,179)

| metric | score |
| --- | --- |
| precision@5 | 1.00 |
| precision@10 | 1.00 |
| accuracy @0.5 | 0.877 |

⚠️ Labels are Gemini 2.5 Flash-generated (no human gold set yet): the scores
measure agreement with the teacher (~90% human agreement per a 50-sample
audit), not ground truth. Complaint-based intent is under-detected (safe
direction for lead queues: misses recall, not purity).

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("google/embeddinggemma-2", device="cpu",
                               model_kwargs={"dtype": torch.float32})
head_path = hf_hub_download("flashcoder/tweet-buy-intent", "head.pt")
head = torch.nn.Sequential(
    torch.nn.Linear(768, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(64, 1))
head.load_state_dict(torch.load(head_path, map_location="cpu"))
head.eval()

with torch.no_grad():
    emb = torch.tensor(embedder.encode(["is there an app for invoicing"], normalize_embeddings=True))
    score = torch.sigmoid(head(emb)).item()
print(score)  # 0.82
```

Trained on 7,866 X posts (2,850 intent) distilled from a Gemini teacher.
Dataset: https://huggingface.co/datasets/flashcoder/tweet-buy-intent-leads
Pipeline: github.com/flashclub/x-lead-intent-pipeline
