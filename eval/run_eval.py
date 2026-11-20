#!/usr/bin/env python3
"""Benchmark runner — score the held-out test set with every configured
adapter, compute teacher-agreement metrics, and emit a comparison report.

Usage:
    cd <repo root>
    python3 eval/run_eval.py --config eval/config.json [--limit 300]

Requires:
    * data/labeled_10k.jsonl (the full labeled dataset)
    * at least one adapter; the local sidecar (:5019) must be running
      for the sidecar adapter
    * pyyaml only if an adapter config uses YAML (JSON config = stdlib)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.adapters import load_adapters
from eval.metrics import evaluate_run


def stratified_split_3way(
    labels: list[bool], seed: int, frac: float = 0.15
) -> tuple[set[int], set[int]]:
    """Exact same split logic as train.py — reproduces the held-out test set."""
    rng = random.Random(seed)
    by_class: dict[bool, list[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)
    val: set[int] = set()
    test: set[int] = set()
    for indices in by_class.values():
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * frac))
        val.update(indices[:n_val])
        test.update(indices[n_val : n_val * 2])
    return val, test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="eval/config.json")
    ap.add_argument("--dataset", default="data/labeled_full_7866.jsonl")
    ap.add_argument("--limit", type=int, default=300, help="stratified eval sample size")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    # load .env (JEV_KEY etc.) — stdlib, no dotenv dependency
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    cfg = json.loads((REPO_ROOT / args.config).read_text(encoding="utf-8"))

    # Reproduce train.py's exact data pipeline: id-dedupe + same filters,
    # so the eval set IS the held-out test set the model was scored on.
    rows = []
    seen: set[str] = set()
    for line in (REPO_ROOT / args.dataset).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = str(r.get("id") or f"row-{len(rows)}")  # anonymized data has no id
        if rid in seen:
            continue
        seen.add(rid)
        label = r.get("label")
        if not isinstance(label, dict) or not isinstance(label.get("intent"), bool):
            continue
        if len(r.get("text", "")) < 20:
            continue
        rows.append(r)
    texts = [r["text"] for r in rows]
    labels = [bool(r["label"]["intent"]) for r in rows]

    _, test_idx = stratified_split_3way(labels, seed=7)
    test_idx = sorted(test_idx)

    # stratified sample of the test set (deterministic)
    rng = random.Random(20261119)
    picked = sorted(rng.sample(test_idx, min(args.limit, len(test_idx))))
    sample_texts = [texts[i] for i in picked]
    sample_labels = [labels[i] for i in picked]
    pos = sum(sample_labels)
    print(f"eval set: {len(sample_texts)} posts ({pos} intent / {len(sample_texts) - pos} noise)")

    adapters = load_adapters(cfg)
    if not adapters:
        print("no adapters configured in", args.config)
        return

    report: dict = {
        "date": str(date.today()),
        "dataset": Path(args.dataset).name,
        "n": len(sample_texts),
        "n_intent": pos,
        "models": {},
    }
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for adapter in adapters:
        name = adapter.name
        print(f"scoring with [{name}] ...")
        try:
            batch = adapter.score(sample_texts)
        except Exception as exc:
            report["models"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            continue
        if len(batch.scores) != len(sample_texts):
            report["models"][name] = {"error": f"score count mismatch ({len(batch.scores)})"}
            continue
        m = evaluate_run(batch.scores, sample_labels, batch.latencies)
        report["models"][name] = m
        lat = m.get("latency", {})
        print(
            f"  acc={m['accuracy']} p@5={m.get('precision_at_5')} p@10={m.get('precision_at_10')} "
            f"ece={m.get('ece')} p50={lat.get('p50_s', '–')}s"
        )

    (out_dir / f"eval-{date.today().isoformat()}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        f"# 评测报告 {date.today().isoformat()}",
        "",
        f"数据：{report['n']} 条（{report['n_intent']} 意向）｜参考基准：teacher 标签（Gemini 2.5 Flash）｜抽样：test 集分层抽样 {len(sample_texts)} 条",
        "",
        "| 模型 | accuracy | precision@5 | precision@10 | ECE | p50 延迟 | p95 延迟 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, m in report["models"].items():
        if "error" in m:
            lines.append(f"| {name} | ERROR: {m['error']} | | | | | |")
            continue
        lat = m.get("latency", {})
        lines.append(
            f"| {name} | {m['accuracy']} | {m.get('precision_at_5')} | "
            f"{m.get('precision_at_10')} | {m.get('ece')} | "
            f"{lat.get('p50_s', '–')}s | {lat.get('p95_s', '–')}s |"
        )
    lines += [
        "",
        "> 口径说明：所有 accuracy/precision 均为 **teacher 一致率**"
        "（teacher 人工一致率约 90%，见 docs/jev-research-2026-11-19.md §5.3），非 ground truth。",
    ]
    (out_dir / f"eval-{date.today().isoformat()}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {out_dir}/eval-{date.today().isoformat()}.md")


if __name__ == "__main__":
    main()
