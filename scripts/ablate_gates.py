"""闸门消融：逐个关掉检索层的判定，看指标掉不掉——**掉才说明它承重**。

为什么需要：闸门是我在真实数据上一条条加的，但从没验证过哪条是真的承重。
如果某条关掉后指标不动，它就是在白增加复杂度。

```bash
python scripts/ablate_gates.py
python scripts/ablate_gates.py --eval-path eval/qa_real.jsonl --json
```

⚠️ 只改运行期对象，跑完自动恢复，不写盘、不改索引。
"""

from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import PROJECT_ROOT, settings
from src.pipeline.ablation import render, run_ablation
from src.pipeline.runtime import IndexMissingError, load_retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索闸门消融实验")
    parser.add_argument("--eval-path", default=str(PROJECT_ROOT / "eval" / "qa_real.jsonl"))
    parser.add_argument(
        "--from-log",
        action="store_true",
        help="改为回放真实查询日志（无标注，比的是行为变化；日志路径见 QUERY_LOG_*）",
    )
    parser.add_argument(
        "--from-corpus",
        action="store_true",
        help="改为回放「语料里真实用户问过的问题」（他人写的，比我自己出的评测集独立）",
    )
    parser.add_argument("--limit", type=int, default=200, help="--from-log / --from-corpus 的上限")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or "WARNING")

    try:
        bundle = load_retriever(settings)
    except IndexMissingError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    if args.from_log or args.from_corpus:
        from src.pipeline.ablation import render_replay, replay_queries

        if args.from_log:
            from src.pipeline.query_log import read_records

            records = read_records(settings.query_log_path, limit=args.limit)
            queries = [str(row.get("query") or "") for row in records]
            source = "查询日志"
        else:
            from src.pipeline.real_queries import mine_queries

            queries = mine_queries(bundle.chunks, limit=args.limit)
            source = "语料里的真实提问"

        if not queries:
            print(f"{source}里没有可回放的查询。先真实用几次（网页界面或 ask.py），再来跑这一项。")
            return 0

        print(f"回放来源：{source}（{len(queries)} 条）\n")
        results = replay_queries(queries, settings, bundle=bundle)
        if args.json:
            print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
        else:
            print(render_replay(results))
        return 0

    try:
        results = run_ablation(args.eval_path, settings, bundle=bundle)
    except Exception as exc:  # noqa: BLE001 - 评测集与语料不匹配等
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
    else:
        print(render(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
