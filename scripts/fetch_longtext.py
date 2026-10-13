"""把已采集语料里**被截断的长博文**补成全文。

```bash
python scripts/fetch_longtext.py --dry-run          # 先看有多少条要补
python scripts/fetch_longtext.py --limit 20         # 先补 20 条试水
python scripts/fetch_longtext.py                    # 全量补全（每条 1 个请求，按 CRAWL_INTERVAL 限速）
python scripts/build_index.py --source raw          # 补完必须增量重建索引
```

为什么需要它：微博的 `container/getIndex` 对长博文只返回截断版（结尾是「… 全文」），
实测 **331/1478（22.4%）** 的采到的博文受影响，且**中位长度 151 字**（未截断的只有 69 字）
——被砍掉的正是信息量最大的那批。取不回全文时会把标记换成显式说明，**不会假装完整**。
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import settings
from src.crawler.weibo_client import WeiboClient, WeiboCrawlError
from src.pipeline.longtext import candidates_in, refresh_long_posts, summarize
from src.storage import load_posts, save_posts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="补全被截断的长博文")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不发请求、不写盘")
    parser.add_argument("--limit", type=int, default=None, help="最多补多少条（试水用）")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or settings.log_level)

    posts = load_posts(settings.raw_posts_path)
    if not posts:
        print(f"[错误] 没有语料：{settings.raw_posts_path}", file=sys.stderr)
        return 2

    targets = candidates_in(posts)
    print(f"语料 {len(posts)} 条，其中被截断 {len(targets)} 条（{len(targets) / len(posts):.1%}）")
    if args.dry_run:
        print("--dry-run：不发请求。去掉它就会开始逐条补全。")
        return 0
    if not targets:
        print("没有需要补全的博文。")
        return 0

    try:
        client = WeiboClient(settings)
    except WeiboCrawlError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    # 分批补 + 每批落盘：全量 331 条要跑十几分钟，中途断掉不能把已补的全丢掉。
    # 每批结束后失败的那些已被标注为「已截断」，不会再被当成候选，所以循环必然收敛。
    batch = max(1, args.limit or 40)
    remaining = posts
    total_fetched = total_failed = 0
    while True:
        refreshed, stats = refresh_long_posts(remaining, client.fetch_long_text, limit=batch)
        if not stats.updated:
            break
        save_posts(settings.raw_posts_path, refreshed)
        remaining = refreshed
        total_fetched += stats.fetched
        total_failed += stats.failed
        print(
            f"  本批：成功 {stats.fetched}，失败 {stats.failed}"
            f"｜累计：成功 {total_fetched}，失败 {total_failed}",
            flush=True,
        )
        if args.limit and total_fetched + total_failed >= args.limit:
            break

    print(f"补全结束：成功 {total_fetched}，失败 {total_failed}")
    print(summarize(stats))
    if total_fetched or total_failed:
        print(f"已写回 {settings.raw_posts_path}")
        print("接着重建索引：python scripts/build_index.py --source raw")
    return 0


if __name__ == "__main__":
    sys.exit(main())
