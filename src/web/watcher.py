"""索引热重载：让「自动更新」真正闭环。

## 要解决的问题

`serve.py` 在**启动时装载一次**索引（加载 BGE 约 5 秒），而每天凌晨的定时任务是在
**另一个进程**里写新索引的。于是：不重启服务，就会一直拿旧索引回答。

实测踩过：夜间更新跑完（1478 → 1509 帖 / 1642 → 1672 片段），页面上的片段数还是旧的，
看起来像「自动更新没生效」。当时的临时办法是在页面上提示「请重启服务」——
但一个需要人工介入的「自动」更新，不算闭环。

## 两种触发都做，因为它们各有缺口

| 方式 | 兜住什么 | 缺口 |
|---|---|---|
| **后台轮询**（本模块的 `IndexWatcher`） | 服务先启动、更新后发生（最主要的情况），不需要任何人访问页面 | 最坏要等一个轮询间隔 |
| **主动通知**（`daily_update` 跑完 POST `/api/reload`） | 把窗口从「最多一个间隔」压到接近 0 | 服务没在跑时无效（那种情况本来也不需要重载） |

## 两条工程约束

1. **重载绝不阻塞请求**：跑在后台线程里，请求侧只做一次属性读取；
2. **切换是一次属性赋值**（Python 里这是原子的）：正在处理的请求继续用完它手里那份
   旧 bundle，不会出现「一半新一半旧」的状态。旧 bundle 由那个请求的局部引用保活，
   不会因为被换掉就失效。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable

from src.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)


def index_signature(settings: Settings | None = None) -> tuple[int, int] | None:
    """索引清单的 ``(mtime_ns, size)``；清单不存在时返回 ``None``。

    为什么盯 manifest 而不是整个索引目录：`build_index` 是**最后**才写 manifest 的，
    所以它的时间戳变化就等于「这次构建结束了」。盯目录 mtime 会在构建过程中反复触发。
    """
    settings = settings or default_settings
    path: Path = settings.manifest_path
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def manifest_summary(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or default_settings
    try:
        return json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


class IndexWatcher:
    """后台盯着索引清单，变了就重载检索器。

    ``loader`` 可注入（默认 ``load_retriever``），因此测试不需要加载真实模型。
    """

    def __init__(
        self,
        app: Any,
        *,
        settings: Settings | None = None,
        interval: float = 30.0,
        loader: Callable[[Settings], Any] | None = None,
    ):
        self.app = app
        self.settings = settings or default_settings
        self.interval = max(1.0, float(interval))
        self._loader = loader
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._signature = index_signature(self.settings)
        self.reloads = 0
        self.failures = 0

    # ------------------------------------------------------------------ 单次检查
    def check_once(self) -> bool:
        """变了就重载并返回 True；没变返回 False。同步、可测。"""
        current = index_signature(self.settings)
        if current is None or current == self._signature:
            return False
        with self._lock:
            # 双重检查：拿锁期间可能已经被另一个触发（轮询 / 通知）重载过了
            current = index_signature(self.settings)
            if current is None or current == self._signature:
                return False
            try:
                bundle = self.reload_now()
            except Exception as exc:  # noqa: BLE001 - 重载失败必须保留旧索引继续服务
                self.failures += 1
                logger.warning("索引重载失败，继续使用旧索引：%s", exc)
                return False
            logger.info("索引已热重载：%s 个片段", len(getattr(bundle, "chunks", []) or []))
            self.reloads += 1
            return True

    def reload_now(self) -> Any:
        """立刻重载并切换（``/api/reload`` 也走这里）。"""
        loader = self._loader
        if loader is None:
            from src.pipeline.runtime import load_retriever

            loader = load_retriever
        bundle = loader(self.settings)
        self.app.state.bundle = bundle          # 原子切换
        self._signature = index_signature(self.settings)
        return bundle

    # ------------------------------------------------------------------ 线程
    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.check_once()
            except Exception:  # noqa: BLE001 - 轮询线程绝不能因为一次异常退出
                logger.exception("索引轮询出现异常（已忽略，下一轮继续）")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="index-watcher", daemon=True)
        self._thread.start()
        logger.info("索引热重载已启动（每 %.0f 秒检查一次清单）", self.interval)

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
