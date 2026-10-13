"""列出语料里的博文——用于人工检查数据质量、以及标注评测集。

用法：
    python scripts/list_posts.py                       # 前 20 条，正文截断
    python scripts/list_posts.py --limit 100 --full    # 全部正文
    python scripts/list_posts.py --search 柄图          # 只看包含关键词的
    python scripts/list_posts.py --source mock          # 看 mock 语料
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import settings
from src.crawler.mock import load_mock_posts
from src.storage import load_posts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="列出语料里的博文")
    parser.add_argument("--limit", type=int, default=20, help="最多显示多少条")
    parser.add_argument("--full", action="store_true", help="显示完整正文")
    parser.add_argument("--search", default="", help="只看包含该关键词的博文")
    parser.add_argument("--source", choices=["raw", "mock"], default="raw")
    parser.add_argument("--ids-only", action="store_true", help="只列 id 与作者")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or "WARNING")

    posts = load_mock_posts() if args.source == "mock" else load_posts(settings.raw_posts_path)
    if args.search:
        posts = [post for post in posts if args.search in post.text or args.search in post.author]

    print(f"语料：{args.source} | 命中 {len(posts)} 条（最多显示 {args.limit} 条）")
    print()
    for index, post in enumerate(posts[: args.limit], start=1):
        meta = " | ".join(part for part in (post.author, post.created_at) if part)
        print(f"[{index}] id={post.id}  {meta}")
        if args.ids_only:
            continue
        body = post.text if args.full else post.text[:160] + ("…" if len(post.text) > 160 else "")
        print(f"    {body}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
