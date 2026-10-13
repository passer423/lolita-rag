"""每日更新编排测试（全离线：注入假客户端与假登录检查器）。

这里最重要的两条断言：
1. **登录态失效时一个采集请求都不发**——这是「失效即停止」策略的核心；
2. **没有新增博文时不重建索引**——否则每晚都要白加载一次 BGE 模型。
"""

from __future__ import annotations

import pytest
import requests

from src.config import Settings
from src.crawler.weibo_client import LoginStatus
from src.pipeline.crawl import load_state
from src.pipeline.daily_update import (
    NotLoggedInError,
    notify_server_reload,
    run_daily_update,
)
from src.schema import Post
from src.storage import load_posts, read_jsonl


class FakeClient:
    def __init__(self, pages: dict[tuple[str, int], list[str]]):
        self.pages = pages
        self.calls: list[tuple[str, int]] = []

    def search_page(self, keyword: str, page: int) -> list[Post]:
        self.calls.append((keyword, page))
        return [
            Post(
                id=post_id,
                text=f"关于柄图与裙撑的测试内容，编号 {post_id}，写得足够长以便成为独立片段。",
                url=f"https://example.com/{post_id}",
            )
            for post_id in self.pages.get((keyword, page), [])
        ]


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        data_dir=tmp_path,
        vector_backend="simple",
        embedding_backend="hash",
        daily_keywords="Lolita,lo裙",
        daily_pages=2,
    )
    base.update(overrides)
    return Settings(**base)


def logged_in() -> LoginStatus:
    return LoginStatus(ok=True, logged_in=True, uid="1", screen_name="测试账号")


def logged_out() -> LoginStatus:
    return LoginStatus(ok=False, logged_in=False, detail="Cookie 已失效")


TWO_KEYWORD_PAGES = {
    ("Lolita", 1): ["a1", "a2"],
    ("Lolita", 2): ["a3"],
    ("lo裙", 1): ["b1"],
    ("lo裙", 2): ["b2"],
}


def test_not_logged_in_aborts_without_any_crawl_request(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(TWO_KEYWORD_PAGES)

    with pytest.raises(NotLoggedInError) as excinfo:
        run_daily_update(settings=settings, client=client, login_checker=logged_out)

    assert "Cookie 已失效" in str(excinfo.value)
    assert client.calls == [], "登录态失效时不应该发出任何采集请求"
    assert not settings.raw_posts_path.exists()
    assert not settings.history_path.exists()


def test_happy_path_collects_and_builds_index(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(TWO_KEYWORD_PAGES)

    report = run_daily_update(settings=settings, client=client, login_checker=logged_in)

    assert client.calls == [("Lolita", 1), ("Lolita", 2), ("lo裙", 1), ("lo裙", 2)]
    assert report.keywords == ["Lolita", "lo裙"]
    assert report.pages == 2
    assert report.account == "测试账号"
    assert report.fetched == 5
    assert report.new_posts == 5
    assert report.total_posts == 5
    assert report.chunks_added == 5
    assert report.embedded == 5
    assert report.total_chunks == 5
    assert len(load_posts(settings.raw_posts_path)) == 5

    rows = read_jsonl(settings.history_path)
    assert len(rows) == 1
    assert rows[0]["new_posts"] == 5
    assert rows[0]["keywords"] == ["Lolita", "lo裙"]


def test_second_run_same_day_skips_index_rebuild(tmp_path):
    """每晚都会重扫（daily 语义），但内容没变时不该重建索引、不该调用模型。"""
    settings = make_settings(tmp_path)
    first = run_daily_update(settings=settings, client=FakeClient(TWO_KEYWORD_PAGES), login_checker=logged_in)

    client = FakeClient(TWO_KEYWORD_PAGES)
    second = run_daily_update(settings=settings, client=client, login_checker=logged_in)

    assert len(client.calls) == 4, "daily 模式每次都要重扫前几页"
    assert second.fetched == 5, "抓取量不变（重复的靠 id 去重）"
    assert second.new_posts == 0
    assert second.chunks_added == 0
    assert second.embedded == 0, "没有新内容就不该调用向量模型"
    assert any("跳过索引更新" in note for note in second.notes)
    assert second.total_chunks == first.total_chunks
    assert len(read_jsonl(settings.history_path)) == 2, "每次运行都要留一条历史"


def test_new_post_next_night_is_incrementally_indexed(tmp_path):
    settings = make_settings(tmp_path)
    run_daily_update(
        settings=settings,
        client=FakeClient({("Lolita", 1): ["a1"], ("lo裙", 1): ["b1"]}),
        login_checker=logged_in,
    )

    report = run_daily_update(
        settings=settings,
        client=FakeClient({("Lolita", 1): ["a1", "a9"], ("lo裙", 1): ["b1"]}),
        login_checker=logged_in,
    )

    assert report.new_posts == 1
    assert report.chunks_added == 1
    assert report.embedded == 1, "只有新博文需要编码"
    assert report.total_posts == 3
    assert report.total_chunks == 3


def test_no_keywords_configured_raises(tmp_path):
    settings = make_settings(tmp_path, daily_keywords="")
    with pytest.raises(ValueError, match="关键词"):
        run_daily_update(settings=settings, client=FakeClient({}), login_checker=logged_in)


def test_daily_cursor_is_recorded_separately_from_backfill(tmp_path):
    settings = make_settings(tmp_path)
    run_daily_update(settings=settings, client=FakeClient(TWO_KEYWORD_PAGES), login_checker=logged_in)

    state = load_state(settings)
    assert state["daily"]["pages"] == 2
    assert state["daily"]["keywords"] == ["Lolita", "lo裙"]
    assert state["daily"]["fetched_total"] == 5
    assert "keywords" not in state or state["keywords"] == {}, (
        "daily 模式不应写 backfill 的 per-keyword 游标，否则会污染补历史进度"
    )


# ------------------------------------------------------------------ 通知服务重载

def test_run_daily_update_does_not_notify_by_default(tmp_path, monkeypatch):
    """库函数默认**不发网络请求**：否则单元测试会去打本机的服务端口（甚至触发真重载）。"""
    calls = []
    monkeypatch.setattr(
        "src.pipeline.daily_update.requests.post",
        lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError("不该发请求")),
    )
    settings = make_settings(tmp_path)

    run_daily_update(settings=settings, client=FakeClient(TWO_KEYWORD_PAGES), login_checker=logged_in)

    assert calls == []


def test_injected_notifier_result_lands_in_history(tmp_path):
    """通知结果写进 history，方便事后判断「服务当时有没有被通知到」。"""
    settings = make_settings(tmp_path)

    run_daily_update(
        settings=settings,
        client=FakeClient(TWO_KEYWORD_PAGES),
        login_checker=logged_in,
        notifier=lambda _settings: "已通知本地服务重载索引",
    )

    rows = read_jsonl(settings.history_path)
    assert "已通知本地服务重载索引" in rows[-1]["notes"]


def test_notify_server_reload_is_fire_and_forget(tmp_path, monkeypatch):
    """服务没开 / 返回 500 都不能让每日更新算失败——索引已经写好在磁盘上了。"""
    settings = make_settings(tmp_path)

    def unreachable(*_args, **_kwargs):
        raise requests.RequestException("连接被拒绝")

    monkeypatch.setattr("src.pipeline.daily_update.requests.post", unreachable)
    assert "未运行" in notify_server_reload(settings)

    class BadResponse:
        status_code = 500
        text = "内部错误"

    monkeypatch.setattr("src.pipeline.daily_update.requests.post", lambda *a, **k: BadResponse())
    assert "HTTP 500" in notify_server_reload(settings)

    class GoodResponse:
        status_code = 200
        text = '{"status":"ok","chunks":1672}'

    monkeypatch.setattr("src.pipeline.daily_update.requests.post", lambda *a, **k: GoodResponse())
    assert "已通知" in notify_server_reload(settings)
