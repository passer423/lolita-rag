"""本地回归：跑测试 + 跑评测，和基线比，**掉指标就退出非零**。

逻辑在 ``src/pipeline/regression.py``（可单测），这里只负责参数与输出。

用法：
    python scripts/regression.py              # 跑一遍，和 eval/baseline.json 比
    python scripts/regression.py --update     # 把当前结果写成新基线
    python scripts/regression.py --skip-tests # 只跑评测（更快）
    python scripts/regression.py --eval-path eval/qa_mock.jsonl   # 用 mock 语料跑（CI 用）

退出码：0 = 没有回退；1 = 有指标下降 / 测试失败；2 = 跑不起来（缺索引等）。
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import PROJECT_ROOT
from src.pipeline.regression import (
    BASELINE_PATH,
    compare,
    load_baseline,
    measure,
    render,
    save_baseline,
)
from src.pipeline.runtime import IndexMissingError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地回归：测试 + 评测 + 基线对比")
    parser.add_argument("--eval-path", default=str(PROJECT_ROOT / "eval" / "qa_real.jsonl"))
    parser.add_argument("--baseline", default=str(BASELINE_PATH))
    parser.add_argument("--update", action="store_true", help="把当前结果写成新基线")
    parser.add_argument("--skip-tests", action="store_true", help="跳过 pytest（只跑评测）")
    parser.add_argument("--skip-answer-eval", action="store_true", help="跳过生成层评测")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or "WARNING")

    try:
        current = measure(
            args.eval_path, skip_tests=args.skip_tests, skip_answer_eval=args.skip_answer_eval
        )
    except IndexMissingError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    baseline = load_baseline(args.baseline)
    print(render(current, baseline))

    if args.update:
        path = save_baseline(current, args.baseline)
        print(f"\n已写入新基线：{path}")
        return 0

    if baseline is None:
        print(f"\n还没有基线（{args.baseline}）。确认当前结果可接受后运行：")
        print("  python scripts/regression.py --update")
        return 0

    regressions, notes = compare(current, baseline)
    for note in notes:
        print(f"\n提示：{note}")
    if not regressions:
        print("\n✅ 没有回退")
        return 0

    print("\n❌ 发现回退：")
    for item in regressions:
        print(f"  · {item}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
