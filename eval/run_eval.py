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
from eval.metrics import evaluate_run, roc_auc


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
        "methodology": {
            "split_reproduction": "train.py 同款三路分层切分（seed=7）还原 held-out test 集",
            "sampling": f"test 集内确定性分层抽样 {len(sample_texts)} 条（rng seed=20261119）",
            "reference": "teacher 标签（Gemini 2.5 Flash）；teacher 人工一致率约 90%（50 条审计）",
            "threshold": 0.5,
            "latency": "sidecar 为服务端 per-item 计时；Jev 为客户端逐请求计时（含网络）",
            "jev_protocol": {
                "endpoint": "POST https://api.typesafe.ai/v1/systemone",
                "model": "jev-latest（实际服务版本见 raw.served_model，当前 jev-1.14.0）",
                "question": "Does this X post express buying intent for a product, tool, app or service?",
                "criteria_true": "the author is looking to buy, adopt, or pay for a product, tool, app or service",
                "criteria_false": "content consumption, chatter, praise, or news with no purchase intent",
                "polarity_check": "强意向帖 0.93 / 噪音帖 0.08（noul = P(buy-intent=true)，极性正确）",
                "mode": "单条逐请求，无批处理",
            },
        },
        "models": {},
    }
    per_sample: dict = {
        "index": picked,
        "text": sample_texts,
        "teacher_label": sample_labels,
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
        m["roc_auc"] = roc_auc(batch.scores, sample_labels)
        report["models"][name] = m
        per_sample[f"score: {name}"] = [round(v, 4) for v in batch.scores]
        per_sample[f"latency_s: {name}"] = [round(v, 4) for v in batch.latencies]
        lat = m.get("latency", {})
        print(
            f"  acc={m['accuracy']} p@5={m.get('precision_at_5')} p@10={m.get('precision_at_10')} "
            f"ece={m.get('ece')} p50={lat.get('p50_s', '–')}s"
        )

    (out_dir / f"eval-{date.today().isoformat()}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── 逐条明细 CSV ──
    import csv
    names = list(report["models"].keys())
    scored = [n for n in names if "error" not in report["models"][n]]
    if scored:
        cols = ["index", "teacher_label", "text"]
        for n in scored:
            cols += [f"score: {n}", f"pred(0.5): {n}", f"correct: {n}", f"latency_s: {n}"]
        csv_path = out_dir / f"eval-{date.today().isoformat()}-samples.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for i in range(len(sample_texts)):
                row = [picked[i], int(sample_labels[i]), sample_texts[i].replace("\n", " ")]
                for n in scored:
                    sc = per_sample[f"score: {n}"][i]
                    pred = sc >= 0.5
                    row += [sc, int(pred), int(pred == sample_labels[i]), per_sample[f"latency_s: {n}"][i]]
                w.writerow(row)

    # ── 分歧分析：两模型预测不一致的样本 ──
    disagreement_lines: list[str] = []
    if len(scored) >= 2:
        a, b = scored[0], scored[1]
        pa = [per_sample[f"score: {n}"][i] >= 0.5 for i, n in [(i, a) for i in range(len(sample_texts))]]
        pb = [per_sample[f"score: {n}"][i] >= 0.5 for i, n in [(i, b) for i in range(len(sample_texts))]]
        n_dis = sum(1 for x, y in zip(pa, pb) if x != y)
        report["disagreement"] = {"models": [a, b], "n": n_dis,
                                  "pct": round(n_dis / len(sample_texts), 3)}
        if n_dis:
            disagreement_lines.append(f"### 分歧样本（{a} vs {b}，共 {n_dis} 条）\n")
            for i in range(len(sample_texts)):
                if pa[i] != pb[i]:
                    ok_a = per_sample[f"score: {a}"][i] >= 0.5 == sample_labels[i]
                    ok_b = per_sample[f"score: {b}"][i] >= 0.5 == sample_labels[i]
                    winner = a if ok_a and not ok_b else (b if ok_b and not ok_a else "平手/双错")
                    t = sample_texts[i].replace("\n", " ")[:80]
                    disagreement_lines.append(
                        f"- [{winner}] ours={per_sample[f'score: {a}'][i]:.2f} "
                        f"jev={per_sample[f'score: {b}'][i]:.2f} teacher={int(sample_labels[i])} ｜ {t}"
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
        "",
        "## 测试方法",
        "",
        f"- **数据**：{report['dataset']}（7,866 条全量标注），按 train.py 同款三路分层切分（seed=7）还原 held-out test 集，再确定性分层抽样 {len(sample_texts)} 条（{report['n_intent']} 意向 / {report['n'] - report['n_intent']} 噪音，rng seed=20261119）",
        "- **参考基准**：teacher 标签（Gemini 2.5 Flash）；teacher 对人工判断一致率约 90%（50 条审计），故本报告所有百分比为 teacher 一致率",
        "- **判定阈值**：0.5；指标含混淆矩阵、precision/recall/F1、ROC-AUC、ECE（10 桶校准）、precision@5/10/25",
        "- **延迟口径**：sidecar = 服务端 per-item 计时；Jev = 客户端逐请求计时（含网络往返）",
        "- **Jev 接入**：POST api.typesafe.ai/v1/systemone（System One 协议，state + noul criteria），模型 jev-latest（实际 jev-1.14.0），单条逐请求；极性验证：强意向帖 0.93 / 噪音帖 0.08",
        "",
        "## 逐条明细",
        "",
        f"完整 300 条 × 每模型（score/pred/correct/latency）见 `eval-{date.today().isoformat()}-samples.csv`（与本报告同目录）。",
    ]
    lines += disagreement_lines
    (out_dir / f"eval-{date.today().isoformat()}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {out_dir}/eval-{date.today().isoformat()}.md")


if __name__ == "__main__":
    main()
