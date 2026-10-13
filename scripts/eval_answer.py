"""生成层评测：看用户真正读到的那段话，而不只是检索结果。

```bash
python scripts/eval_answer.py                    # 真实语料 + 当前生成配置
python scripts/eval_answer.py --limit 5          # 先跑几条看看
python scripts/eval_answer.py --json             # 机器可读，便于回归对比
python scripts/eval_answer.py --problem-limit 0  # 列出全部有问题的答案
```

没有 ``LLM_API_KEY`` 时也能跑：生成会退化为抽取式答案，本脚本照样给出
「兜底路径本身是否干净」的数字。**注意两组数字不可混为一谈**——
抽取式答案天然不会编造，等配上 Key 跑真模型，`疑似事实越界` 才会变成有意义的指标。
"""

from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import settings
from src.pipeline.answer_eval import evaluate_answers
from src.pipeline.runtime import IndexMissingError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成层（答案质量）评测")
    parser.add_argument("--eval-path", default=None, help="评测集 jsonl（默认 eval/qa_real.jsonl）")
    parser.add_argument("--top-k", type=int, default=None, help="检索条数（默认取配置 TOP_K）")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 题")
    parser.add_argument("--problem-limit", type=int, default=10, help="列出多少条问题答案（0 = 全部）")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="额外做语义忠实度评审（LLM-as-judge，每题多一次模型调用）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or settings.log_level)
    eval_path = args.eval_path or str(settings.data_dir.parent / "eval" / "qa_real.jsonl")

    judge = None
    if args.judge:
        from src.generator.judge import AnswerJudge

        judge = AnswerJudge(settings)
        if not settings.llm_enabled:
            print("[错误] --judge 需要配置 LLM_API_KEY", file=sys.stderr)
            return 2

    try:
        report = evaluate_answers(
            eval_path, settings, top_k=args.top_k, limit=args.limit, judge=judge
        )
    except IndexMissingError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"[错误] 找不到评测集：{exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(report.render())
    if not settings.llm_enabled:
        print()
        print(
            "提示：当前没有配置 LLM_API_KEY，答案走的是抽取式兜底——它天然不会编造，"
            "所以「疑似事实越界」必然是 0。配上 Key 再跑一次，这个数字才有意义。"
        )
    if args.problem_limit != 10:
        problems = report.problems
        shown = problems if args.problem_limit == 0 else problems[: args.problem_limit]
        print()
        print(f"有问题的答案（共 {len(problems)} 条，列出 {len(shown)} 条）：")
        for item in shown:
            print(f"  · {item.question}")
            for problem in item.problems:
                print(f"      - {problem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
