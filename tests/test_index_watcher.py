"""索引热重载的测试。

这个功能的价值是「自动更新真正闭环」，但它的**风险**是：一个后台线程拿着锁去加载模型，
一旦写错就会把正在服务的进程弄坏。所以测试重心是三条底线：

1. **重载失败必须保留旧索引**——模型下载失败/内存不足时，服务不能变成不可用；
2. **切换是原子的**——请求侧只会看到「旧的」或「新的」，没有中间态；
3. **管理动作只许本机**——`/api/reload` 会重载模型，不能因为 `--host 0.0.0.0`
   就变成局域网任何人都能触发。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.web.app import create_app
from src.web.watcher import IndexWatcher, index_signature


def write_manifest(settings: Settings, chunks: int) -> None:
    settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.write_text(
        json.dumps({"chunks": chunks, "embedder": "x"}), encoding="utf-8"
    )


def fake_bundle(chunks: int):
    return SimpleNamespace(
        retriever=SimpleNamespace(bm25=SimpleNamespace(min_ratio=0.2, document_frequency=lambda t: 0)),
        chunks=[f"c{i}" for i in range(chunks)],
        backend="simple",
    )


# ------------------------------------------------------------------ 签名

def test_signature_is_none_without_manifest(tmp_path):
    assert index_signature(Settings(data_dir=tmp_path)) is None


def test_signature_changes_when_manifest_is_rewritten(tmp_path):
    settings = Settings(data_dir=tmp_path)
    write_manifest(settings, 10)
    first = index_signature(settings)
    assert first is not None

    # 内容与大小都不同 → 签名必须变（只比 mtime 在某些文件系统上会漏掉）
    write_manifest(settings, 99999)
    assert index_signature(settings) != first


# ------------------------------------------------------------------ 监视器

def make_watcher(tmp_path, chunks_now: int, loaded: dict):
    settings = Settings(data_dir=tmp_path)
    write_manifest(settings, chunks_now)
    app = SimpleNamespace(state=SimpleNamespace(bundle=fake_bundle(chunks_now)))

    def loader(_settings):
        loaded["calls"] = loaded.get("calls", 0) + 1
        return fake_bundle(loaded.get("chunks", 100))

    watcher = IndexWatcher(app, settings=settings, loader=loader)
    return watcher, app, settings


def test_check_once_does_nothing_when_manifest_unchanged(tmp_path):
    watcher, app, _ = make_watcher(tmp_path, 10, {})
    before = app.state.bundle

    assert watcher.check_once() is False
    assert app.state.bundle is before, "没变化就不该碰 bundle"


def test_check_once_reloads_after_manifest_changes(tmp_path):
    watcher, app, settings = make_watcher(tmp_path, 10, {"chunks": 100})
    write_manifest(settings, 150)          # 模拟夜间更新写完新索引

    assert watcher.check_once() is True
    assert len(app.state.bundle.chunks) == 100
    assert watcher.reloads == 1


def test_reload_failure_keeps_the_old_bundle(tmp_path):
    """底线一：模型加载失败时服务必须照常可用，只是还在用旧索引。"""
    settings = Settings(data_dir=tmp_path)
    write_manifest(settings, 10)
    old = fake_bundle(10)
    app = SimpleNamespace(state=SimpleNamespace(bundle=old))

    def broken_loader(_settings):
        raise RuntimeError("模型文件损坏")

    watcher = IndexWatcher(app, settings=settings, loader=broken_loader)
    write_manifest(settings, 99999)

    assert watcher.check_once() is False
    assert app.state.bundle is old, "重载失败不能把服务搞成没有索引"
    assert watcher.failures == 1


def test_reload_failure_does_not_retry_forever_within_one_signature(tmp_path):
    """失败后签名不推进，下一轮还会试——这是刻意的（暂时性故障应当能自愈）。"""
    settings = Settings(data_dir=tmp_path)
    write_manifest(settings, 10)
    app = SimpleNamespace(state=SimpleNamespace(bundle=fake_bundle(10)))
    attempts = {"n": 0}

    def sometimes(_settings):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("第一次失败")
        return fake_bundle(42)

    watcher = IndexWatcher(app, settings=settings, loader=sometimes)
    write_manifest(settings, 99999)

    assert watcher.check_once() is False
    assert watcher.check_once() is True, "第二轮应当重试并成功"
    assert len(app.state.bundle.chunks) == 42


def test_watcher_thread_starts_and_stops(tmp_path):
    watcher, _, _ = make_watcher(tmp_path, 10, {})
    watcher.interval = 1.0
    watcher.start()
    watcher.start()          # 重复 start 不应起第二个线程
    assert watcher._thread is not None
    watcher.stop()
    assert watcher._thread is None


# ------------------------------------------------------------------ 接口

def test_reload_endpoint_rejects_non_loopback_clients(tmp_path):
    """底线三：`--host 0.0.0.0` 时不能让局域网任何人都能触发重载。"""
    settings = Settings(data_dir=tmp_path, llm_api_key="")
    client = TestClient(create_app(bundle=fake_bundle(2), settings=settings))
    with client:
        # TestClient 默认的 client 是 "testclient"（非本机），应当被拒
        assert client.post("/api/reload").status_code == 403


def test_reload_endpoint_works_for_loopback(tmp_path, monkeypatch):
    """本机调用应当成功重载（这里把加载器换成假的，不真的去加载模型）。

    注意要 patch 的是 **runtime 模块里的 `load_retriever`**：watcher 是在 `reload_now`
    里惰性 import 它的，patch 到别处（比如 `src.web.app._load`）不会生效——
    第一次写这条用例就栽在这儿。
    """
    settings = Settings(data_dir=tmp_path, llm_api_key="")
    loaded = fake_bundle(7)
    monkeypatch.setattr("src.web.app._load", lambda _app: fake_bundle(1))
    monkeypatch.setattr("src.pipeline.runtime.load_retriever", lambda _settings: loaded)

    app = create_app(bundle=None, settings=settings)
    client = TestClient(app, client=("127.0.0.1", 12345))
    with client:
        response = client.post("/api/reload")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "chunks": 7}


def test_injected_bundle_disables_the_watcher(tmp_path):
    """注入了 bundle 的场景（测试）不能启动轮询：它会把注入的假检索器换掉。"""
    settings = Settings(data_dir=tmp_path, llm_api_key="")
    client = TestClient(create_app(bundle=fake_bundle(2), settings=settings))
    with client:
        assert client.app.state.watcher is None


@pytest.mark.parametrize("enabled", [True, False])
def test_watcher_started_only_when_enabled(tmp_path, enabled, monkeypatch):
    # 关掉 lifespan 里的真实装载：临时目录里没有索引，加载必然失败
    monkeypatch.setattr("src.web.app._load", lambda _app: fake_bundle(3))
    settings = Settings(data_dir=tmp_path, llm_api_key="", index_watch_enabled=enabled)
    app = create_app(bundle=None, settings=settings)
    with TestClient(app):
        watcher = app.state.watcher
        assert (watcher is not None) is enabled
        if watcher is not None:
            assert watcher._thread is not None, "进入 lifespan 后线程应当已启动"
