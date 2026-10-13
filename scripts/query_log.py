"""分析查询日志：看用户在搜什么、什么搜不到。

为什么需要它：`data/logs/queries.jsonl` 是**唯一不受「自己出评测题」影响**的证据来源。
`--refused` 列出的每一条，都是「语料/检索没兜住」的候选，直接拿去补语料或补规则。

用法：
    python scripts/query_log.py                    # 概览
    python scripts/query_log.py --refused          # 只列拒答查询（bad case 候选池）
    python scripts/query_log.py --popular 20       # 高频查询
    python scripts/query_log.py --source web       # 只看网页端的查询
    python scripts/query_log.py --json             # 机器可读
"""

from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import settings
from src.pipeline.query_log import read_records, summarize


def _render(summary: dict, path: str) -> str:
    lines = [
        f"日志文件      : {path}",
        f"查询总数      : {summary['total']}",
        f"其中拒答      : {summary['refused']}（{summary['refusal_rate']:.1%}）",
        f"平均命中条数  : {summary['avg_hits']}",
        f"延迟 p50/p95  : {summary['latency_ms']['p50']} / {summary['latency_ms']['p95']} ms"
        f"（最大 {summary['latency_ms']['max']}）",
    ]
    if summary["by_source"]:
        lines.append(
            "来源分布      : " + "、".join(f"{k} {v}" for k, v in summary["by_source"].items())
        )
    if summary["popular"]:
        lines.append("")
        lines.append("高频查询：")
        for query, count in summary["popular"][:20]:
            lines.append(f"  ×{count:<3} {query}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="查询日志分析")
    parser.add_argument("--path", default=None, help="日志路径（默认 data/logs/queries.jsonl）")
    parser.add_argument("--source", default=None, help="只看某一路（web / cli）")
    parser.add_argument("--refused", action="store_true", help="只列拒答查询")
    parser.add_argument("--popular", type=int, default=0, help="只列前 N 个高频查询")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or "WARNING")
    path = args.path or str(settings.query_log_path)
    records = read_records(path)
    summary = summarize(records, source=args.source)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.refused:
        queries = summary["refused_queries"]
        print(f"拒答查询 {len(queries)} 条（这类是 bad case 的主要来源，值得逐条看）：")
        for query in queries:
            print(f"  · {query}")
        if not queries:
            print("  （还没有拒答记录）")
        return 0

    if args.popular:
        print(f"高频查询前 {args.popular}：")
        for query, count in summary["popular"][: args.popular]:
            print(f"  ×{count:<4} {query}")
        return 0

    if not records:
        print(f"还没有查询日志：{path}")
        print("跑一次检索就会产生，例如：python scripts/ask.py \"柄图是什么\"")
        return 0

    print(_render(summary, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
