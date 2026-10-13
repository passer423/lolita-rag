"""从 HAR 抓包里取出完整 Cookie 写入 .env，并可选立刻验证。

为什么用它：手工从 F12 复制 Cookie 很容易漏字段（实测漏过 SSOLoginState / _T_WM，
导致服务端只认游客）。HAR 里存的是浏览器发出去的**完整** Cookie 请求头。

而且**存一次 HAR 就同时搞定两件事**：语料（import_weibo.py）+ Cookie（本脚本）。

用法：
    python scripts/extract_cookie.py --input capture.har                    # 只看，不写
    python scripts/extract_cookie.py --input capture.har --write            # 写入 .env
    python scripts/extract_cookie.py --input capture.har --write --verify   # 写入并立刻验证

输出**不会显示 Cookie 内容**，只显示长度与字段名。
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import PROJECT_ROOT, Settings
from src.crawler.cookie_from_har import extract_cookie_from_har, write_cookie_to_env
from src.crawler.weibo_client import check_login

EXIT_OK = 0
EXIT_NOT_FOUND = 2
EXIT_NOT_LOGGED_IN = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 HAR 抓包提取 Cookie 并写入 .env")
    parser.add_argument("--input", required=True, help="浏览器导出的 .har 文件")
    parser.add_argument("--write", action="store_true", help="写入 .env 的 WEIBO_COOKIE")
    parser.add_argument("--verify", action="store_true", help="写入后立刻用 1 个请求验证登录态")
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"), help="要写入的 .env 路径")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or "WARNING")

    result = extract_cookie_from_har(args.input)
    print(result.render())
    if not result.found:
        print()
        print("没有找到 Cookie。请确认：")
        print("  1. 是在**已登录**状态下导出的 HAR")
        print("  2. 导出前 Network 面板里确实有 weibo 的请求（先翻一下页）")
        print("  3. 用的是 Chrome 的 *Save all as HAR with content*")
        return EXIT_NOT_FOUND

    _, missing = result.present_missing()
    if missing:
        print(f"提示：{'、'.join(missing)} 未出现。若验证失败，换个请求再导出一次 HAR。")

    if not args.write:
        print()
        print("加 --write 写入 .env：")
        print(f"  python scripts/extract_cookie.py --input {args.input} --write --verify")
        return EXIT_OK

    replaced = write_cookie_to_env(args.env, result.cookie)
    print()
    print(f"已{'替换' if replaced else '新增'} {args.env} 里的 WEIBO_COOKIE（{len(result.cookie)} 字符）")

    if not args.verify:
        print("接着运行：python scripts/check_login.py")
        return EXIT_OK

    status = check_login(Settings(weibo_cookie=result.cookie))
    print()
    print(status.render())
    hint = status.cookie_hint()
    if hint:
        print(hint)
    if status.logged_in:
        print()
        print("接下来可以采集了：")
        print("  python scripts/crawl.py --keyword lo裙 --keyword JSK --pages 5")
        print("  python scripts/build_index.py --source raw")
    return EXIT_OK if status.logged_in else EXIT_NOT_LOGGED_IN


if __name__ == "__main__":
    sys.exit(main())
