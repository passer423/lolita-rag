"""每日定时更新入口（供 Windows 任务计划器调用）。

用法：
    python scripts/daily_update.py                     # 关键词/页数取自 .env
    python scripts/daily_update.py --keyword lo裙 --pages 3
    python scripts/daily_update.py --skip-login-check  # 仅调试用

退出码（任务计划器可据此判断成败）：
    0  成功
    2  配置 / 采集 / 索引错误
    3  Cookie 失效——需要你手动更新 .env 里的 WEIBO_COOKIE
    4  已有实例在运行，本次跳过

关于「自动登录」：本项目**不做**用账号密码自动登录或绕验证码，
登录始终由人在浏览器里完成（见 README 的说明）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import settings
from src.crawler.weibo_client import LoginStatus, WeiboCrawlError
from src.pipeline.daily_update import (
    EXIT_ALREADY_RUNNING,
    EXIT_ERROR,
    EXIT_NOT_LOGGED_IN,
    EXIT_OK,
    NotLoggedInError,
    notify_server_reload,
    run_daily_update,
)
from src.pipeline.lock import AlreadyRunningError, single_instance_lock

logger = logging.getLogger("lolita_rag.daily_update")


def _report_failure(message: str) -> None:
    """失败既要打印给人看，也要写进日志文件。

    这个任务凌晨无人值守运行，只在 stderr 上打印等于没有记录——
    第二天你只会看到退出码，却不知道原因。
    """
    logger.error(message)
    print(message, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="每日定时更新（采集 + 增量建索引）")
    parser.add_argument("--keyword", action="append", help="覆盖 DAILY_KEYWORDS")
    parser.add_argument("--pages", type=int, default=None, help="覆盖 DAILY_PAGES")
    parser.add_argument(
        "--skip-login-check",
        action="store_true",
        help="跳过登录态预检（仅调试用；正式运行不要加）",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    settings.ensure_dirs()
    log_file = settings.logs_dir / f"update-{datetime.now():%Y%m%d-%H%M%S}.log"
    setup_logging(args.log_level or settings.log_level, log_file=log_file)
    print(f"日志文件：{log_file}")

    login_checker = None
    if args.skip_login_check:
        login_checker = lambda: LoginStatus(  # noqa: E731 - 只在调试分支使用
            ok=True, logged_in=True, detail="已跳过登录态预检（--skip-login-check）"
        )

    try:
        with single_instance_lock(
            settings.lock_path,
            stale_after=settings.daily_lock_stale_hours * 3600,
        ):
            report = run_daily_update(
                args.keyword,
                settings,
                pages=args.pages,
                login_checker=login_checker,
                # 真实通知由入口接上：库函数默认不发网络请求（否则单元测试会打到本机服务）
                notifier=notify_server_reload,
            )
    except AlreadyRunningError as exc:
        _report_failure(f"[跳过] {exc}")
        return EXIT_ALREADY_RUNNING
    except NotLoggedInError as exc:
        _report_failure(f"[需要操作] {exc}")
        _report_failure(
            "请打开 https://m.weibo.cn/ 确认已登录，重新复制完整 Cookie 写入 .env 的 "
            "WEIBO_COOKIE，然后重跑本命令。"
        )
        return EXIT_NOT_LOGGED_IN
    except (WeiboCrawlError, FileNotFoundError, ValueError) as exc:
        _report_failure(f"[失败] {exc}")
        return EXIT_ERROR

    print(report.render())
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
