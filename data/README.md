# Data

The full 7,866-post dataset is **not** committed (X ToS + user privacy):
rebuild it with `collect_bulk.py` + `label.py`.

`sample_labeled_240.jsonl` — 240-row anonymized sample (handles/ids stripped),
stratified: 90 intent / 150 noise.

Schema:
```json
{"text": "post text", "label": {"intent": true, "type": "recommend_request", "confidence": 1.0}}
```
