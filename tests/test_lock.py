"""单实例锁测试。

锁这种东西只有在「真的并发/残留」时才起作用，所以这里显式构造三种场景：
正常持有、重复获取、陈旧锁接管。
"""

from __future__ import annotations

import os
import time

import pytest

from src.pipeline.lock import AlreadyRunningError, single_instance_lock


def test_lock_is_created_then_released(tmp_path):
    path = tmp_path / "run.lock"
    with single_instance_lock(path):
        assert path.exists()
        assert "pid" in path.read_text(encoding="utf-8")
    assert not path.exists()


def test_second_acquire_is_rejected(tmp_path):
    path = tmp_path / "run.lock"
    with single_instance_lock(path):
        with pytest.raises(AlreadyRunningError):
            with single_instance_lock(path):
                pass  # pragma: no cover - 不应该进来


def test_lock_is_released_when_body_raises(tmp_path):
    path = tmp_path / "run.lock"
    with pytest.raises(RuntimeError):
        with single_instance_lock(path):
            raise RuntimeError("模拟崩溃")
    assert not path.exists(), "异常退出也必须释放锁，否则定时任务会被永久卡死"


def test_stale_lock_is_taken_over(tmp_path):
    path = tmp_path / "run.lock"
    path.write_text('{"pid": 999999}', encoding="utf-8")
    old = time.time() - 10 * 3600
    os.utime(path, (old, old))

    with single_instance_lock(path, stale_after=6 * 3600):
        assert path.exists()
    assert not path.exists()


def test_fresh_lock_is_not_stolen(tmp_path):
    path = tmp_path / "run.lock"
    path.write_text('{"pid": 123}', encoding="utf-8")
    with pytest.raises(AlreadyRunningError, match="已有实例"):
        with single_instance_lock(path, stale_after=6 * 3600):
            pass  # pragma: no cover


def test_missing_parent_directories_are_created(tmp_path):
    path = tmp_path / "nested" / "deeper" / "run.lock"
    with single_instance_lock(path):
        assert path.exists()
