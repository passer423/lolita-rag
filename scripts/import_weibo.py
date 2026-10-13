"""导入手工保存的微博响应 / 页面（**不需要 Cookie**）。

微博的关键词检索接口必须要登录态，所以如果你想自己采集又不想把 Cookie
交出来（或不想让它留在聊天记录里），走这条路：

    1. 浏览器登录 https://m.weibo.cn/ ，搜索你想要的词
    2. F12 → Network → 找到 m.weibo.cn/api/container/getIndex 那条请求
       → 右键 → Copy → Copy response → 粘进一个 .json 文件
       （或者直接 Ctrl+S 保存整个页面为 .html，两种都支持）
    3. 运行本脚本导入

用法：
    python scripts/import_weibo.py --input saved/ --dry-run     # 先只解析，看识别情况
    python scripts/import_weibo.py --input saved/
    python scripts/build_index.py --source raw                  # 导入完重建索引

解析失败时不会静默丢弃，而是把「看到的结构」打印出来——把那段发我即可补适配。
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import settings
from src.crawler.importer import import_paths
from src.pipeline.crawl import merge_and_save


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导入保存好的微博响应或页面")
    parser.add_argument("--input", nargs="+", required=True, help="文件或目录，可传多个")
    parser.add_argument("--dry-run", action="store_true", help="只解析并预览，不写入语料")
    parser.add_argument("--preview", type=int, default=3, help="预览条数")
    parser.add_argument(
        "--exclude",
        default="",
        help="按关键词排除（逗号分隔，支持全角逗号）。常用：出物,抽奖,上新,图透,预约,转发",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or settings.log_level)

    exclude = [
        word.strip()
        for word in args.exclude.replace("，", ",").split(",")
        if word.strip()
    ]
    result = import_paths(args.input, exclude=exclude)
    print(result.render())
    print()

    if not result.posts:
        print(
            "没有解析出任何博文。请把上面的 [kind] 与「说明」内容发给我，我据此补适配。",
            file=sys.stderr,
        )
        return 2

    for post in result.posts[: args.preview]:
        print(f"  [{post.id}] {post.author or '未知'} | {post.text[:64]}")

    if args.dry_run:
        print(f"\n--dry-run：未写入 {settings.raw_posts_path}")
        return 0

    before, after = merge_and_save(result.posts, settings)
    print(f"\n已写入 {settings.raw_posts_path}")
    print(f"  原有 {before} 条 → 现在 {after} 条（新增 {after - before} 条）")
    print("\n接着运行：python scripts/build_index.py --source raw")
    return 0


if __name__ == "__main__":
    sys.exit(main())
