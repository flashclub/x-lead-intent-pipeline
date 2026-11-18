# Data

## `labeled_full_7866.jsonl` — the full labeled dataset (anonymized)

All **7,866 X posts** collected via 27 query families, labeled by a Gemini
2.5 Flash teacher (see `label.py` for the exact prompt and anchors).

Anonymization: tweet id / author handle / follower count / timestamps are
stripped. Post **text** is kept verbatim (it is the model input) together
with labels and provenance. If you are an author of one of these posts and
want yours removed, open an issue — takedowns are honored immediately.

Schema:
```json
{
  "text": "post text (verbatim)",
  "bucket": "intent | noise (the query family that collected it)",
  "source_query": "the X search operator that retrieved it",
  "label": {"intent": true, "type": "recommend_request", "confidence": 1.0}
}
```

Distribution: 2,850 intent / 5,016 noise. Types: recommend_request 2,006 ·
noise 3,534 · praise 1,151 · feature_wish 736 · complaint 341 ·
alternative_seek 98.

Known label caveats (audited on 50 hand-checked samples, see
`data/spot_check_50.md`): teacher agreement ~90% with human judgment;
complaint-based intent is under-detected; engagement-bait prompts that
merely *contain* intent phrasing are correctly rejected.

## `sample_labeled_240.jsonl`

240-row anonymized sample — same schema, for a quick look without cloning
the full file.
