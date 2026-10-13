"""跑离线评测。

示例：
    python scripts/eval.py
    python scripts/eval.py --eval-path eval/qa_mock.jsonl --top-k 5 --json
"""

from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import settings
from src.pipeline.eval import DEFAULT_EVAL_PATH, EvalSetMismatchError, evaluate
from src.pipeline.runtime import IndexMissingError, load_retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索与拒答评测")
    parser.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH), help="评测集 jsonl 路径")
    parser.add_argument("--top-k", type=int, default=None, help="检索返回条数（默认取配置 TOP_K）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本消费")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or settings.log_level)

    try:
        bundle = load_retriever(settings)
    except IndexMissingError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    try:
        report = evaluate(args.eval_path, settings, top_k=args.top_k, bundle=bundle)
    except EvalSetMismatchError as exc:
        print(f"[评测集不匹配] {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
