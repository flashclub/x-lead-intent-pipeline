"""Evaluation metrics — teacher-agreement, ranking and latency.

All metrics are computed against the TEACHER labels (Gemini 2.5 Flash).
Per the project's evaluation policy, these measure teacher agreement,
not ground-truth accuracy (see docs/jev-research-2026-11-19.md §5.3).
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence


def classification(scores: Sequence[float], labels: Sequence[bool], threshold: float = 0.5) -> dict:
    preds = [s >= threshold for s in scores]
    tp = sum(1 for p, y in zip(preds, labels) if p and y)
    fp = sum(1 for p, y in zip(preds, labels) if p and not y)
    fn = sum(1 for p, y in zip(preds, labels) if not p and y)
    tn = sum(1 for p, y in zip(preds, labels) if not p and not y)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    accuracy = (tp + tn) / len(labels) if labels else None
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def precision_at_k(scores: Sequence[float], labels: Sequence[bool], k: int) -> float | None:
    if not scores or k <= 0:
        return None
    k = min(k, len(scores))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    hits = sum(1 for i in ranked if labels[i])
    return hits / k


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Rank-based AUC (Mann-Whitney U). Handles ties via average ranks."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    ranked = sorted(scores)
    # average rank (1-based) with ties
    def avg_rank(v: float) -> float:
        lo = ranked.index(v)  # first occurrence
        hi = len(ranked) - 1 - ranked[::-1].index(v)  # last occurrence
        return (lo + hi) / 2 + 1
    rank_sum = sum(avg_rank(s) for s in pos)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def expected_calibration_error(
    scores: Sequence[float], labels: Sequence[bool], bins: int = 10
) -> dict:
    """ECE + per-bin reliability data (10 equal-width bins over [0,1])."""
    if not scores:
        return {"ece": None, "bins": []}
    bucket: dict[int, list[tuple[float, bool]]] = {i: [] for i in range(bins)}
    for s, y in zip(scores, labels):
        b = min(bins - 1, int(s * bins))
        bucket[b].append((s, 1.0 if y else 0.0))
    ece = 0.0
    bin_rows = []
    n = len(scores)
    for i in range(bins):
        group = bucket[i]
        if not group:
            continue
        conf = sum(g[0] for g in group) / len(group)
        acc = sum(g[1] for g in group) / len(group)
        gap = abs(acc - conf)
        ece += (len(group) / n) * gap
        bin_rows.append(
            {"bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}", "n": len(group),
             "confidence": round(conf, 4), "accuracy": round(acc, 4), "gap": round(gap, 4)}
        )
    return {"ece": round(ece, 4), "bins": bin_rows}


def latency_stats(latencies: Sequence[float]) -> dict:
    if not latencies:
        return {}
    ordered = sorted(latencies)
    n = len(ordered)
    return {
        "mean_s": round(sum(ordered) / n, 4),
        "p50_s": round(ordered[int(0.5 * (n - 1))], 4),
        "p95_s": round(ordered[min(n - 1, int(0.95 * (n - 1)))], 4),
        "max_s": round(ordered[-1], 4),
        "n": n,
    }


def evaluate_run(
    scores: Sequence[float], labels: Sequence[bool], latencies: Sequence[float]
) -> dict:
    out: dict = {"n": len(scores)}
    out.update(classification(scores, labels))
    for k in (5, 10, 25):
        out[f"precision_at_{k}"] = precision_at_k(scores, labels, k)
    out.update(expected_calibration_error(scores, labels))
    out.update({"latency": latency_stats(latencies)})
    return out
