"""pytest 引导：让测试可以 ``from src.xxx import ...``，并**隔离真实 data/ 目录**。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True, scope="session")
def _isolate_real_data_dir(tmp_path_factory):
    """任何测试都不许写进项目真实的 ``data/``。

    实测踩过这个坑：查询日志默认开启，跑一次全套测试就往真实日志里灌了几十条——
    拒答率、延迟分位这些统计被自己的测试污染，而那份日志本来是用来观察**真实使用**的。

    只改 ``settings`` 单例**不够**：测试里显式构造的 ``Settings(...)`` 会重新读配置，
    仍然指向真实 ``data/``。所以三层一起堵：

    1. 设环境变量 ``DATA_DIR`` / ``QUERY_LOG_ENABLED``（新构造的 ``Settings`` 会读到）；
    2. 改默认单例的 ``data_dir`` / ``query_log_enabled``（用单例的调用方也覆盖）；
    3. 需要测日志的用例自己构造 ``Settings(data_dir=tmp_path)``，不受影响。
    """
    from src.config import settings as default_settings

    tmp = tmp_path_factory.mktemp("lolita-rag-data")
    env_backup = {key: os.environ.get(key) for key in ("DATA_DIR", "QUERY_LOG_ENABLED")}
    os.environ["DATA_DIR"] = str(tmp)
    os.environ["QUERY_LOG_ENABLED"] = "false"

    original_dir = default_settings.data_dir
    original_log = default_settings.query_log_enabled
    default_settings.data_dir = tmp
    default_settings.query_log_enabled = False
    try:
        yield tmp
    finally:
        default_settings.data_dir = original_dir
        default_settings.query_log_enabled = original_log
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

