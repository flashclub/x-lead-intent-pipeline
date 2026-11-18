# Tweet Buy-Intent Miner

End-to-end pipeline that mines **buyer-intent posts** from X (Twitter) and
distills an LLM teacher into a tiny, fast, on-device classifier.

> "Is there an app that …", "anyone recommend …", "I wish there was …",
> "alternative to …" — thousands of people announce their buying intentions
> on X every day. This pipeline finds them, scores them, and serves the
> scores over HTTP in **~0.3s per batch on a MacBook CPU**.

```
┌─────────┐    ┌──────────┐    ┌─────────────────┐    ┌───────────────┐
│ collect │ →  │  label   │ →  │     train       │ →  │     serve     │
│ X API   │    │ LLM teach│    │ frozen embed +  │    │ FastAPI :5019 │
│ 7,866   │    │ Gemini   │    │ MLP head (CPU)  │    │ POST /score   │
└─────────┘    └──────────┘    └─────────────────┘    └───────────────┘
```

## Results (held-out test set, n=1,179 — never seen by early stopping)

| metric | score |
| --- | --- |
| precision@5 | 1.00 |
| precision@10 | 1.00 |
| accuracy @0.5 | 0.877 |

**Honest read**: labels are LLM-generated (Gemini 2.5 Flash teacher), so these
numbers measure *agreement with the teacher*, whose own agreement with human
judgment is ~90% (audited on 50 hand-checked samples). The teacher's one
systematic bias — complaint-based intent is under-detected — is inherited by
the model. Top-of-queue precision is unaffected (the bias misses recall, not
purity), which is exactly what a lead-queue product needs.

## Pipeline

| step | script | what it does |
| --- | --- | --- |
| 1. collect | `collect_bulk.py` | 27 query families (15 intent / 12 hard-negative), deep pagination, resume-safe, ~7.9k posts in 40 min |
| 2. label | `label.py` | Gemini teacher via OpenAI-compatible API, infinite 429 backoff (never drops a post), 4 workers, resume-safe |
| 3. train | `train.py` | frozen `embeddinggemma-2` (768-d) → MLP 768-64-1, mini-batch + early stopping on val, **three-way stratified split** (test never touched), all on CPU fp32 |
| 4. serve | `serve.py` | FastAPI `POST /score {"texts": [...]} → {scores, elapsed_sec}` |
| bonus | `benchmark.py` | A/B embedders on your own data — measured beats cited |

`collect.py` is the original single-page version, kept for a quick 100-post smoke run.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. collect (needs SCRAPER_API_KEY for an X data provider)
SCRAPER_API_KEY=... python3 collect_bulk.py

# 2. label (needs an OpenAI-compatible teacher endpoint)
MULERUN_API_KEY=... python3 label.py --in data/posts_10k.jsonl --out data/labeled.jsonl --workers 2

# 3. train (~1-3h encoding on CPU; the MLP head itself trains in minutes)
python3 train.py --data data/labeled.jsonl --out model

# 4. serve
python3 serve.py   # POST http://127.0.0.1:5019/score {"texts": ["..."]}
```

A pre-trained head (`model/head.pt`, 199 KB) + `model/metrics.json` are
included so step 3 is optional: the embedder downloads from Hugging Face on
first run (~1.5 GB), everything else runs on CPU.

**`data/labeled_full_7866.jsonl`** — the full labeled dataset (3.9 MB,
anonymized: handles/ids stripped, text verbatim) is committed, so every
number in this README is reproducible from the repo without scraping.
`data/sample_labeled_240.jsonl` is a small quick-look subset.

## API

```bash
curl -X POST http://127.0.0.1:5019/score \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["is there an app for invoicing", "watched a movie yesterday"]}'
# → {"scores": [0.8256, 0.0021], "elapsed_sec": 0.369, "model": "google/embeddinggemma-2"}
```

## Design notes

- **Why a frozen embedder + tiny head?** The trainable surface is 49k params —
  trains in minutes on CPU, generalizes from ~7k examples, and the whole
  scorer is a 200 KB artifact + a shared embedder. No GPU required at
  training or inference time.
- **Why distill an LLM teacher?** Labeling 10k posts with a frontier LLM API
  costs a few dollars and hours; a human team costs weeks. The trade is
  documented above: you inherit the teacher's biases, so an audit sample is
  mandatory — the audit harness (`spot-check` workflow in this repo's history)
  is part of the pipeline.
- **U+2028 gotcha** (cost us 12 posts): X posts contain U+2028/U+2029 line
  separators. `str.splitlines()` splits on them and silently corrupts JSONL
  reads — every reader in this repo iterates file objects (`\n` only). See
  label.py's loader.
- **429 handling**: the labeler retries rate-limited calls with capped
  exponential backoff forever — a post is never dropped because the teacher
  was busy.

## Limitations

- English only (query plans and teacher prompt are en)
- Labels are teacher-generated; a human-verified gold set is the next
  milestone (see `label.py` audit workflow)
- X data provider ToS applies to collected content — the sample in
  `data/` is anonymized (handles/ids stripped)
- precision@k = 1.00 partly reflects query-conditioned data (intent queries
  retrieve posts with explicit marker phrases); treat it as teacher-agreement,
  not ground-truth precision

## License

MIT for the code. Collected post content belongs to its authors.
