"""检查 WEIBO_COOKIE 是否仍是有效登录会话。

只花 **1 个请求**（``GET m.weibo.cn/api/config``），不采集任何数据。
换 Cookie 后、注册定时任务前，建议先跑这个。

用法：
    python scripts/check_login.py

退出码：0 = 登录态有效；3 = 无效或未配置
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

from src.config import settings
from src.crawler.weibo_client import check_login


def main() -> int:
    if not settings.weibo_cookie.strip():
        print("未配置 WEIBO_COOKIE：请先复制 .env.example 为 .env 并填入 Cookie。")
        return 3

    status = check_login(settings)
    print(status.render())
    if status.present:
        print(f"常见字段已出现：{'、'.join(status.present)}")
    hint = status.cookie_hint()
    if hint:
        print(hint)

    if status.logged_in:
        print("接下来可以运行：")
        print("  python scripts/daily_update.py            # 先手动跑一次每日更新")
        print("  python scripts/crawl.py --keyword lo裙 --pages 3   # 或者直接批量采集")
        print("  python scripts/register_task.py --time 02:00       # 确认无误后再注册定时任务")
        return 0

    print()
    print(status.detail)
    print()
    print("获取 Cookie 的三种方式（按可靠性排序）：")
    print("  1) 从 HAR 抓包提取（最稳，字段一定完整）")
    print("     F12 → Network → 翻页 → 右键 → Save all as HAR with content")
    print("     python scripts/extract_cookie.py --input capture.har --write")
    print("     python scripts/check_login.py")
    print("  2) F12 → Network → 任意 m.weibo.cn/api/... 请求 → Request Headers")
    print("     → 复制**整个** cookie 的值 → 写入 .env 的 WEIBO_COOKIE=")
    print("  3) Application → Cookies → 全选复制（最容易漏字段，不推荐）")
    print()
    print("前提：浏览器里必须真的处于登录状态（能看到自己的头像）。")
    return 3


if __name__ == "__main__":
    sys.exit(main())
