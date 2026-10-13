"""精排（reranker）效果实测：同一评测集上，开与关的差异。

```bash
python scripts/rerank_eval.py                    # 用 .env 里的 RERANKER_MODEL
python scripts/rerank_eval.py --model data/models/bge-reranker-base
python scripts/rerank_eval.py --json
```

为什么单独一个脚本而不并进回归：精排要额外加载约 1GB 的 cross-encoder，CPU 上每个查询
多花几百毫秒。回归是「本机一条命令、30 秒内跑完」的东西，不该被它拖成依赖模型下载。

它**只重排已经通过闸门的候选**（见 retriever/reranker.py），所以这里预期看到的是：
Recall 不变、拒答不变、**MRR 上升**。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import PROJECT_ROOT, settings
from src.pipeline.eval import evaluate
from src.pipeline.runtime import IndexMissingError, load_retriever
from src.retriever.reranker import Reranker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="精排效果实测")
    parser.add_argument("--eval-path", default=str(PROJECT_ROOT / "eval" / "qa_real.jsonl"))
    parser.add_argument("--model", default=None, help="覆盖 RERANKER_MODEL（可为本地目录）")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or "WARNING")
    if args.model:
        settings.reranker_model = args.model

    try:
        bundle = load_retriever(settings)
    except IndexMissingError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    # ---------- 关 ----------
    settings.reranker_enabled = False
    settings_before = time.perf_counter()
    before = evaluate(args.eval_path, settings, top_k=args.top_k, bundle=bundle)
    before_seconds = time.perf_counter() - settings_before

    # ---------- 开 ----------
    reranker = Reranker(settings, model=None)
    load_started = time.perf_counter()
    model = reranker._get_model()
    load_seconds = time.perf_counter() - load_started
    if model is None:
        print(f"[错误] 精排模型不可用：{settings.reranker_model}", file=sys.stderr)
        print("提示：HF 主站国内不通，用 modelscope 直链下到本地再指向它（README 有说明）", file=sys.stderr)
        return 2

    settings.reranker_enabled = True
    bundle.retriever.reranker = reranker
    after_started = time.perf_counter()
    after = evaluate(args.eval_path, settings, top_k=args.top_k, bundle=bundle)
    after_seconds = time.perf_counter() - after_started

    # 哪些题的排序变了
    changed = []
    for old, new in zip(before.results, after.results):
        if old.reciprocal_rank != new.reciprocal_rank or old.refused != new.refused:
            changed.append(
                {
                    "question": old.question,
                    "rr_before": round(old.reciprocal_rank, 4),
                    "rr_after": round(new.reciprocal_rank, 4),
                    "refused_before": old.refused,
                    "refused_after": new.refused,
                }
            )

    payload = {
        "model": settings.reranker_model,
        "before": {
            "recall_at_k": round(before.recall_at_k, 4),
            "mrr": round(before.mrr, 4),
            "hit_rate": round(before.hit_rate, 4),
            "refusal_accuracy": before.refusal_accuracy,
            "seconds": round(before_seconds, 2),
        },
        "after": {
            "recall_at_k": round(after.recall_at_k, 4),
            "mrr": round(after.mrr, 4),
            "hit_rate": round(after.hit_rate, 4),
            "refusal_accuracy": after.refusal_accuracy,
            "seconds": round(after_seconds, 2),
        },
        "model_load_seconds": round(load_seconds, 2),
        "changed_questions": changed,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    def fmt(value):
        return "—" if value is None else f"{value:.3f}"

    print(f"精排模型：{settings.reranker_model}（加载 {load_seconds:.1f} 秒）")
    print(f"评测集  ：{args.eval_path}（{before.total} 题，top_k={settings.top_k}）\n")
    print(f"{'指标':<16}{'关闭精排':>12}{'开启精排':>12}{'变化':>10}")
    for key, label in (
        ("recall_at_k", "Recall@k"),
        ("mrr", "MRR"),
        ("hit_rate", "Hit@k"),
        ("refusal_accuracy", "拒答准确率"),
    ):
        old_value = payload["before"][key]
        new_value = payload["after"][key]
        delta = "—" if old_value is None or new_value is None else f"{new_value - old_value:+.4f}"
        print(f"{label:<16}{fmt(old_value):>12}{fmt(new_value):>12}{delta:>10}")
    print(
        f"{'总耗时(秒)':<16}{payload['before']['seconds']:>12.2f}"
        f"{payload['after']['seconds']:>12.2f}"
    )
    per_query = (after_seconds - before_seconds) / max(before.total, 1) * 1000
    print(f"\n每题平均额外开销：{per_query:.0f} 毫秒（CPU）")

    print(f"\n排序发生变化的题：{len(changed)} / {before.total}")
    for item in changed[:10]:
        arrow = "↑" if item["rr_after"] > item["rr_before"] else "↓"
        print(
            f"  {arrow} rr {item['rr_before']:.3f} → {item['rr_after']:.3f}  {item['question'][:38]}"
        )
    improved = sum(1 for item in changed if item["rr_after"] > item["rr_before"])
    worsened = sum(1 for item in changed if item["rr_after"] < item["rr_before"])
    print(f"  其中变好 {improved} 题，变差 {worsened} 题")
    return 0


if __name__ == "__main__":
    sys.exit(main())
