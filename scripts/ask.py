"""提问（检索 + 生成）或只做检索。

示例：
    # 检索 + 生成；没配 LLM_API_KEY 时自动退化为「原文摘录」，检索照样能用
    python scripts/ask.py "带亮片的裙子怎么洗？"

    # 只看检索结果，完全不调用生成层（推荐用来看语料里到底有没有相关内容）
    python scripts/ask.py "柄图" --retrieval-only
    python scripts/ask.py "柄图" --retrieval-only --top-k 10

    # 机器可读，便于脚本消费或做回归对比
    python scripts/ask.py "尺码怎么选" --retrieval-only --json

    # 交互式连续提问（输入 exit / q 退出）
    python scripts/ask.py --interactive
"""

from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import settings
from src.pipeline.ask import ask, hits_to_dicts, render_answer, render_hits, retrieve
from src.pipeline.runtime import IndexMissingError, load_retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索语料并回答问题")
    parser.add_argument("question", nargs="?", help="要查询的问题或关键词")
    parser.add_argument("--top-k", type=int, default=None, help="返回条数（默认取配置 TOP_K）")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只输出检索结果，不调用生成层",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    parser.add_argument("--interactive", action="store_true", help="进入交互式问答")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or settings.log_level)

    try:
        bundle = load_retriever(settings)
    except IndexMissingError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    mode_label = "仅检索（不生成）" if args.retrieval_only else "检索 + 生成"

    def handle(question: str) -> None:
        if args.retrieval_only:
            hits = retrieve(question, settings, top_k=args.top_k, bundle=bundle)
            if args.json:
                print(json.dumps(hits_to_dicts(hits), ensure_ascii=False, indent=2))
            else:
                print(render_hits(hits, question))
            return

        answer = ask(question, settings, top_k=args.top_k, bundle=bundle)
        if args.json:
            print(
                json.dumps(
                    {
                        "question": question,
                        "used_llm": answer.used_llm,
                        "note": answer.note,
                        "answer": answer.text,
                        "citations": answer.citations,
                        "hits": hits_to_dicts(answer.hits),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(render_answer(answer))

    if args.interactive or not args.question:
        print(f"进入交互模式（{mode_label}）。直接输入问题，输入 exit / q 退出。")
        while True:
            try:
                question = input("\n问> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit", "q"}:
                break
            if question:
                handle(question)
        return 0

    handle(args.question)
    return 0


if __name__ == "__main__":
    sys.exit(main())
