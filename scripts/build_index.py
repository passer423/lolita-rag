"""构建 / 更新检索索引。

示例：
    python scripts/build_index.py                # 增量更新（默认，只编码新内容）
    python scripts/build_index.py --rebuild      # 全量重建
    python scripts/build_index.py --source raw   # 用 data/raw/posts.jsonl
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  （导入即完成 sys.path 设置）
from _bootstrap import setup_logging

from src.config import settings
from src.pipeline.build_index import build_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 lolita-rag 检索索引")
    parser.add_argument("--source", choices=["mock", "raw"], default="mock", help="数据来源（默认 mock）")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="全量重建（清空向量库）；不加则走增量，只编码新增或变更的片段",
    )
    parser.add_argument("--max-len", type=int, default=400, help="单个片段最大字符数")
    parser.add_argument("--overlap", type=int, default=50, help="相邻片段的字符重叠")
    parser.add_argument(
        "--min-len",
        type=int,
        default=None,
        help=f"短于该字数的博文不入库（默认取配置 CHUNK_MIN_LEN={settings.chunk_min_len}）",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or settings.log_level)
    stats = build_index(
        source=args.source,
        rebuild=args.rebuild,
        max_len=args.max_len,
        overlap=args.overlap,
        min_len=args.min_len,
    )
    print(stats.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
