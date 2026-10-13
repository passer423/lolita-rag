"""单实例锁。

为什么需要：定时任务可能在上一次还没跑完时被再次触发（机器休眠后唤醒、
手动补跑、时钟漂移）。两个采集进程同时跑会立刻把请求频率翻倍，
既容易触发风控，也让「每页落盘」的进度账本互相覆盖。

实现用 ``O_CREAT | O_EXCL`` 原子创建，而不是「先判断存在再写」——
后者在两次调用之间留了竞态窗口。锁内容写入 pid 与开始时间，便于排查。
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

DEFAULT_STALE_SECONDS = 6 * 3600


class AlreadyRunningError(RuntimeError):
    """已有实例持有锁。"""


@contextmanager
def single_instance_lock(
    path: Path | str,
    stale_after: float = DEFAULT_STALE_SECONDS,
) -> Iterator[Path]:
    """持有锁期间执行代码块；退出（含异常）时释放。

    ``stale_after`` 秒之前的锁被视为陈旧（进程被 kill -9 或断电时不会执行
    finally，锁文件会残留），允许接管，避免定时任务被永久卡死。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            age = 0.0
        if age <= stale_after:
            holder = path.read_text(encoding="utf-8", errors="replace")[:200]
            raise AlreadyRunningError(
                f"已有实例在运行（锁 {path}，已持有 {age / 60:.1f} 分钟）：{holder}"
            )
        logger.warning(
            "发现陈旧锁（已 %.1f 小时 > %.1f 小时），接管：%s",
            age / 3600,
            stale_after / 3600,
            path,
        )
        try:
            path.unlink()
        except OSError as exc:  # 删不掉就照常尝试创建，交由 O_EXCL 判定
            logger.warning("陈旧锁删除失败（%s），继续尝试创建", exc)

    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise AlreadyRunningError(f"已有实例在运行（锁 {path}）") from exc

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(
            {"pid": os.getpid(), "started_at": datetime.now().isoformat(timespec="seconds")},
            handle,
            ensure_ascii=False,
        )
    try:
        yield path
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("锁释放失败：%s（%s）", path, exc)
