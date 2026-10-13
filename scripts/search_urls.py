"""生成 s.weibo.com 搜索链接清单，配合「一次 HAR 导出」批量扩充语料。

这是**不需要 Cookie** 的最快路径：

    1. 运行本脚本，拿到一串搜索链接
    2. 在浏览器里逐个打开（可以中键批量打开，或用 --open 代劳）
    3. 确认页面都加载完 → F12 → Network → 右键 → *Save all as HAR with content*
    4. python scripts/import_weibo.py --input capture.har --exclude "出物,抽奖,转发,上新,图透"

一次导出就包含所有页面的响应，不必一页一页 Ctrl+S。

用法：
    python scripts/search_urls.py                                  # 内置词表 x 5 页
    python scripts/search_urls.py --keywords "lo裙,JSK" --pages 3
    python scripts/search_urls.py --open --pages 2                 # 直接开标签页（会开很多）
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.crawler.search_urls import DEFAULT_KEYWORDS, PROMO_EXCLUDE, build_search_urls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成微博搜索链接清单，配合 HAR 批量采集")
    parser.add_argument("--keywords", default="", help="逗号分隔；留空用内置词表")
    parser.add_argument("--pages", type=int, default=5, help="每个关键词翻几页（默认 5）")
    parser.add_argument("--open", action="store_true", help="直接用默认浏览器打开（会开很多标签，慎用）")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or "WARNING")

    if args.keywords.strip():
        keywords = [
            word.strip()
            for word in args.keywords.replace("，", ",").split(",")
            if word.strip()
        ]
    else:
        keywords = list(DEFAULT_KEYWORDS)

    urls = build_search_urls(keywords, args.pages)
    print(f"关键词 {len(keywords)} 个 x {args.pages} 页 = {len(urls)} 个链接")
    print()
    for index, url in enumerate(urls, start=1):
        print(f"{index:>3}. {url}")
    print()
    print("接下来：")
    print("  1. 逐个打开这些链接（确认页面加载完成）")
    print("  2. F12 → Network → 右键 → Save all as HAR with content")
    print(f'  3. python scripts/import_weibo.py --input capture.har --exclude "{PROMO_EXCLUDE}"')
    print()
    print("提示：分页的相邻页常有重复，导入时会自动去重，不用担心。")

    if args.open:
        import webbrowser

        print(f"\n正在打开 {len(urls)} 个标签页……")
        for url in urls:
            webbrowser.open_new_tab(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
