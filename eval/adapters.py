"""Adapters — every scorable model speaks the same interface.

DecisionModel.score(texts) -> ScoreBatch

ScoreBatch:
    scores: list[float]        0-1 per text (higher = more buy-intent)
    latencies: list[float]     per-text inference seconds (measured)
    raw: list[dict]            adapter-specific metadata (optional)

Adapters included:
    SidecarAdapter  — local FastAPI sidecar (intent/serve.py, :5019)
    JevApiAdapter   — Jev / System-One compatible API (config-driven;
                      endpoint/key/question are configurable, the adapter
                      normalizes any probability response to 0-1 scores)
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScoreBatch:
    scores: list[float]
    latencies: list[float]
    raw: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.scores) > 0


class DecisionModel(ABC):
    name: str

    @abstractmethod
    def score(self, texts: list[str]) -> ScoreBatch:
        """Score texts. Must return one score per text, or raise."""


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _dig(obj: Any, path: str) -> Any:
    """Dot-path lookup into nested dicts: 'data.0.score' or 'probability'."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


class SidecarAdapter(DecisionModel):
    """Local FastAPI sidecar (intent/serve.py). per-item latency mode."""

    def __init__(self, name: str, base_url: str, timeout: int = 120):
        self.name = name
        self.base_url = base_url.rstrip("/")

    def score(self, texts: list[str]) -> ScoreBatch:
        t0 = time.perf_counter()
        data = _post_json(
            f"{self.base_url}/score",
            {"texts": texts, "per_item": True},
            {},
            timeout=120,
        )
        scores = [float(v) for v in data["scores"]]
        # per-text latency measured server-side
        latencies = [float(v) for v in data.get("times_sec", [])] or [
            (time.perf_counter() - t0) / len(texts)
        ] * len(texts)
        raw = [{"source": "sidecar"} for _ in texts]
        return ScoreBatch(scores=scores, latencies=latencies, raw=raw)


class JevApiAdapter(DecisionModel):
    """Jev / System-One compatible API adapter (config-driven).

    Follows the public Jev API pattern: send {text, question}, receive a
    probability per answer option. The adapter normalizes P(yes) to a
    0-1 buy-intent score. Endpoint/key/question come from config.yaml.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key_env: str,
        model: str,
        question: str,
        probability_path: str = "probability",
        timeout: int = 60,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.question = question
        self.probability_path = probability_path
        self.timeout = timeout
        self.api_key = os.environ.get(api_key_env, "")

    CRITERIA = {
        "true": "the author is looking to buy, adopt, or pay for a product, tool, app or service",
        "false": "content consumption, chatter, praise, or news with no purchase intent",
    }

    def score(self, texts: list[str]) -> ScoreBatch:
        """Real TypeSafe System One protocol (verified 2026-11-20):

        POST {base_url}/systemone
        {"model": ..., "state": <text>,
         "questions": {"buy_intent": {"type": "noul", "question": ...,
                       "criteria": {"true": ..., "false": ...}}}}
        -> {"answers": {"buy_intent": {"type": "noul", "noul": 0.93}}}

        `noul` is P(buy-intent=true) — polarity verified: strong-intent
        post scores 0.93, noise post 0.08. One text per request.
        """
        scores: list[float] = []
        latencies: list[float] = []
        raw: list[dict] = []
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for text in texts:
            t0 = time.perf_counter()
            data = _post_json(
                f"{self.base_url}/systemone",
                {
                    "model": self.model,
                    "state": text[:1500],
                    "questions": {
                        "buy_intent": {
                            "type": "noul",
                            "question": self.question,
                            "criteria": self.CRITERIA,
                        }
                    },
                },
                headers,
                timeout=self.timeout,
            )
            prob = float(_dig(data, self.probability_path))  # answers.buy_intent.noul
            scores.append(prob)
            latencies.append(time.perf_counter() - t0)
            raw.append({"source": self.base_url, "served_model": data.get("model")})
        return ScoreBatch(scores=scores, latencies=latencies, raw=raw)


def load_adapters(cfg: dict) -> list[DecisionModel]:
    """Instantiate adapters from a config dict."""
    models: list[DecisionModel] = []
    for entry in cfg.get("adapters", []):
        kind = entry.get("type")
        if kind == "sidecar":
            models.append(SidecarAdapter(entry["name"], entry["base_url"]))
        elif kind == "jev":
            models.append(
                JevApiAdapter(
                    name=entry["name"],
                    base_url=entry["base_url"],
                    api_key_env=entry.get("api_key_env", "JEV_API_KEY"),
                    model=entry.get("model", "jev-1"),
                    question=entry.get(
                        "question",
                        "Does this X post express buying intent for a product, "
                        "tool, app or service? Answer yes or no.",
                    ),
                    probability_path=entry.get("probability_path", "probability"),
                    timeout=entry.get("timeout", 60),
                )
            )
        else:
            raise ValueError(f"unknown adapter type: {kind}")
    return models

