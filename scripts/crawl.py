"""采集真实微博数据（支持断点续爬）。

示例：
    # 先只预览解析结果，不写盘、不动进度
    python scripts/crawl.py --keyword Lolita --pages 1 --dry-run

    # 正式采集；中断后重跑同一条命令会从上次的页码继续
    python scripts/crawl.py --keyword Lolita --keyword 洛丽塔 --pages 3

    # 继续上次采集（关键词从进度文件读取）
    python scripts/crawl.py --pages 6

    # 忽略已保存的进度，从头重采
    python scripts/crawl.py --keyword Lolita --pages 3 --restart
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import settings
from src.crawler.weibo_client import WeiboCrawlError
from src.pipeline.crawl import crawl, load_state


def _print_state() -> None:
    state = load_state(settings)
    entries = state.get("keywords") or {}
    if not entries:
        print("采集进度：暂无记录")
        return
    print("采集进度：")
    for keyword, entry in entries.items():
        print(
            f"  「{keyword}」已采 {entry.get('collected', 0)} 条，"
            f"下次从第 {entry.get('next_page', 1)} 页继续（更新于 {entry.get('updated_at')}）"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="采集微博语料")
    parser.add_argument(
        "--keyword",
        action="append",
        help="关键词，可重复传入；省略则续采进度文件中记录的关键词",
    )
    parser.add_argument("--pages", type=int, default=None, help="每个关键词采到第几页（默认取配置）")
    parser.add_argument("--restart", action="store_true", help="忽略已保存的进度，从第 1 页重采")
    parser.add_argument("--dry-run", action="store_true", help="只预览解析结果，不写盘也不更新进度")
    parser.add_argument("--preview", type=int, default=3, help="dry-run 时预览条数")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or settings.log_level)
    _print_state()

    try:
        posts = crawl(
            args.keyword,
            settings,
            pages=args.pages,
            dry_run=args.dry_run,
            preview=args.preview,
            resume=not args.restart,
        )
    except WeiboCrawlError as exc:
        print(f"[采集失败] {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[参数错误] {exc}", file=sys.stderr)
        return 2

    print(f"{'预览' if args.dry_run else '采集'}完成，本次拿到 {len(posts)} 条。")
    if args.dry_run:
        print("确认解析正常后，去掉 --dry-run 重新运行即可写盘。")
    else:
        print("接着运行：python scripts/build_index.py --source raw")
    return 0


if __name__ == "__main__":
    sys.exit(main())
