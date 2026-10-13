"""启动本地检索界面。

用法：
    python scripts/serve.py                     # http://127.0.0.1:8000
    python scripts/serve.py --port 8765
    python scripts/serve.py --host 0.0.0.0      # 允许局域网访问（默认只监听本机）

按 Ctrl+C 停止。索引在启动时装载一次（BGE 约 5 秒），之后每次检索都是毫秒级；
**运行期间索引变化会自动热重载**（每 INDEX_WATCH_INTERVAL 秒检查一次清单，
`daily_update.py` 跑完还会主动调 `/api/reload`），所以每天凌晨更新之后不需要手动重启。
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动 lolita-rag 本地检索界面")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认只允许本机访问）")
    parser.add_argument(
        "--port",
        type=int,
        default=settings.web_port,
        help="端口（默认取 WEB_PORT，即 8000；daily_update 也按这个端口通知重载）",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or settings.log_level)

    import uvicorn

    from src.web.app import create_app

    print(f"检索界面： http://{args.host}:{args.port}")
    print("接口文档： http://%s:%s/docs" % (args.host, args.port))
    if settings.index_watch_enabled:
        print(f"索引热重载： 已开启（每 {settings.index_watch_interval:.0f} 秒检查一次清单）")
    print("按 Ctrl+C 停止。")
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
