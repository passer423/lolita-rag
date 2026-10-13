"""脚本公共引导：把项目根目录放进 sys.path，并统一日志格式。

放在这里的原因：直接执行 ``python scripts/ask.py`` 时，``sys.path[0]`` 是 scripts/
目录而不是项目根目录，不处理的话 ``import src.*`` 会失败。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """配置日志。

    ``log_file`` 会额外写一份到文件：定时任务在凌晨无人值守运行，
    没有文件日志就等于出了问题也查不到。文件用 utf-8 打开——
    语料里的 emoji 在 GBK 下会触发 UnicodeEncodeError。
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # 用 utf-8-sig（带 BOM）：本机只有 Windows PowerShell 5.1，
        # 它的 Get-Content 会按 GBK 解码无 BOM 的 UTF-8，中文日志全是乱码。
        # 注意：只对人读的日志这么做；jsonl 等机器格式必须保持无 BOM，
        # 否则 json.loads 会因行首的 BOM 直接报错。
        handlers.append(logging.FileHandler(log_file, encoding="utf-8-sig"))
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    _make_stdout_never_crash()


def _make_stdout_never_crash() -> None:
    """让 stdout/stderr 遇到无法编码的字符时降级为 '?' 而不是抛异常。

    必要原因：Windows 中文控制台默认是 cp936（GBK）。微博语料和模型答案里
    随时可能出现 emoji 或生僻字，默认行为会让 print 直接抛 UnicodeEncodeError，
    整个命令以 traceback 结束——这属于「输出失败拖垮业务逻辑」，必须避免。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except Exception:  # noqa: BLE001 - 环境不支持时静默跳过
                pass
