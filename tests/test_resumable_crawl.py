"""断点续爬测试。

用假客户端注入，完全不碰网络。重点验证三件事：
1. 中途失败后，下一次运行从中断的页码继续，而不是从头重采；
2. 每页都落盘，中断最多丢一页；
3. 重复采集同一页是幂等的，不会产生重复博文。
"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.pipeline.crawl import crawl, load_state
from src.schema import Post
from src.storage import load_posts


class FakeClient:
    """假客户端：按 (关键词, 页码) 返回预置博文，并记录调用顺序。"""

    def __init__(self, pages: dict[tuple[str, int], list[str]], fail_on: set[tuple[str, int]] | None = None):
        self.pages = pages
        self.fail_on = fail_on or set()
        self.calls: list[tuple[str, int]] = []

    def search_page(self, keyword: str, page: int) -> list[Post]:
        self.calls.append((keyword, page))
        if (keyword, page) in self.fail_on:
            raise RuntimeError("模拟网络中断")
        return [
            Post(id=post_id, text=f"关于柄图和裙撑的测试内容，第 {post_id} 条。", url=f"https://example.com/{post_id}")
            for post_id in self.pages.get((keyword, page), [])
        ]


def make_settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, vector_backend="simple", embedding_backend="hash")


def test_crawl_resumes_from_saved_page_after_failure(tmp_path):
    settings = make_settings(tmp_path)
    pages = {("Lolita", 1): ["a1"], ("Lolita", 2): ["a2"], ("Lolita", 3): ["a3"]}

    broken = FakeClient(pages, fail_on={("Lolita", 2)})
    with pytest.raises(RuntimeError):
        crawl(["Lolita"], settings, pages=3, client=broken)

    # 第 1 页已经落盘，进度指向第 2 页
    state = load_state(settings)
    assert state["keywords"]["Lolita"]["next_page"] == 2
    assert [post.id for post in load_posts(settings.raw_posts_path)] == ["a1"]

    resumed = FakeClient(pages)
    crawl(["Lolita"], settings, pages=3, client=resumed)

    assert resumed.calls[0] == ("Lolita", 2), "应当从中断处继续，而不是重采第 1 页"
    assert resumed.calls == [("Lolita", 2), ("Lolita", 3)]
    assert [post.id for post in load_posts(settings.raw_posts_path)] == ["a1", "a2", "a3"]
    assert load_state(settings)["keywords"]["Lolita"]["next_page"] == 4


def test_increasing_pages_continues_instead_of_restarting(tmp_path):
    settings = make_settings(tmp_path)
    pages = {("Lolita", p): [f"b{p}"] for p in range(1, 4)}

    crawl(["Lolita"], settings, pages=1, client=FakeClient(pages))
    assert load_state(settings)["keywords"]["Lolita"]["next_page"] == 2

    client = FakeClient(pages)
    crawl(["Lolita"], settings, pages=3, client=client)
    assert client.calls == [("Lolita", 2), ("Lolita", 3)]


def test_restart_ignores_saved_progress(tmp_path):
    settings = make_settings(tmp_path)
    pages = {("Lolita", p): [f"c{p}"] for p in range(1, 3)}

    crawl(["Lolita"], settings, pages=1, client=FakeClient(pages))
    client = FakeClient(pages)
    crawl(["Lolita"], settings, pages=2, client=client, resume=False)
    assert client.calls == [("Lolita", 1), ("Lolita", 2)]


def test_keywords_default_to_state_for_continuing(tmp_path):
    settings = make_settings(tmp_path)
    pages = {("洛丽塔", p): [f"d{p}"] for p in range(1, 3)}

    crawl(["洛丽塔"], settings, pages=1, client=FakeClient(pages))
    client = FakeClient(pages)
    # 不传关键词 → 从进度文件里取出「洛丽塔」继续
    posts = crawl(None, settings, pages=2, client=client)
    assert client.calls == [("洛丽塔", 2)]
    assert [post.id for post in posts] == ["d2"]


def test_no_keywords_anywhere_raises(tmp_path):
    settings = make_settings(tmp_path)
    with pytest.raises(ValueError):
        crawl(None, settings, pages=1, client=FakeClient({}))


def test_empty_page_marks_keyword_finished_and_stops(tmp_path):
    settings = make_settings(tmp_path)
    pages = {("Lolita", 1): ["e1"]}          # 第 2 页返回空
    client = FakeClient(pages)
    crawl(["Lolita"], settings, pages=5, client=client)

    assert client.calls == [("Lolita", 1), ("Lolita", 2)], "遇到空页应当停止翻页"
    assert load_state(settings)["keywords"]["Lolita"]["next_page"] == 2


def test_repeat_page_is_idempotent(tmp_path):
    settings = make_settings(tmp_path)
    pages = {("Lolita", 1): ["f1", "f2"]}

    crawl(["Lolita"], settings, pages=1, client=FakeClient(pages))
    crawl(["Lolita"], settings, pages=1, client=FakeClient(pages), resume=False)

    ids = [post.id for post in load_posts(settings.raw_posts_path)]
    assert sorted(ids) == ["f1", "f2"], "同一页重采不应产生重复博文"


def test_dry_run_does_not_write_anything(tmp_path):
    settings = make_settings(tmp_path)
    pages = {("Lolita", 1): ["g1"]}
    client = FakeClient(pages)

    posts = crawl(["Lolita"], settings, pages=1, dry_run=True, client=client)

    assert [post.id for post in posts] == ["g1"]
    assert not settings.raw_posts_path.exists()
    assert not settings.crawl_state_path.exists()
    assert client.calls == [("Lolita", 1)]


def test_skip_when_already_past_requested_pages(tmp_path):
    settings = make_settings(tmp_path)
    pages = {("Lolita", p): [f"h{p}"] for p in range(1, 4)}

    crawl(["Lolita"], settings, pages=3, client=FakeClient(pages))
    client = FakeClient(pages)
    posts = crawl(["Lolita"], settings, pages=3, client=client)

    assert client.calls == [], "已采完的关键词不应再发请求"
    assert posts == []


# --------------------------------------------------------------------- daily 模式
def test_daily_mode_rescans_from_page_one_and_preserves_backfill_cursor(tmp_path):
    """两种采集语义必须能共存：daily 抓新内容，backfill 补历史，游标互不干扰。"""
    settings = make_settings(tmp_path)
    pages = {("Lolita", p): [f"x{p}"] for p in range(1, 4)}

    crawl(["Lolita"], settings, pages=1, client=FakeClient(pages))
    assert load_state(settings)["keywords"]["Lolita"]["next_page"] == 2

    client = FakeClient(pages)
    crawl(["Lolita"], settings, pages=2, mode="daily", client=client)

    assert client.calls == [("Lolita", 1), ("Lolita", 2)], "daily 必须从第 1 页重扫"
    state = load_state(settings)
    assert state["keywords"]["Lolita"]["next_page"] == 2, "daily 不得改动 backfill 的游标"
    assert state["daily"]["pages"] == 2
    assert state["daily"]["last_run"]


def test_daily_mode_restarts_from_page_one_on_every_run(tmp_path):
    settings = make_settings(tmp_path)
    pages = {("Lolita", p): [f"y{p}"] for p in range(1, 3)}

    first = FakeClient(pages)
    crawl(["Lolita"], settings, pages=2, mode="daily", client=first)
    second = FakeClient(pages)
    crawl(["Lolita"], settings, pages=2, mode="daily", client=second)

    assert first.calls == second.calls == [("Lolita", 1), ("Lolita", 2)]


def test_daily_mode_repeat_crawl_is_idempotent(tmp_path):
    settings = make_settings(tmp_path)
    pages = {("Lolita", 1): ["z1", "z2"]}

    crawl(["Lolita"], settings, pages=1, mode="daily", client=FakeClient(pages))
    crawl(["Lolita"], settings, pages=1, mode="daily", client=FakeClient(pages))

    ids = sorted(post.id for post in load_posts(settings.raw_posts_path))
    assert ids == ["z1", "z2"], "每晚重扫不应产生重复博文"


def test_daily_keywords_can_come_from_state(tmp_path):
    settings = make_settings(tmp_path)
    pages = {("lo裙", 1): ["w1"]}

    crawl(["lo裙"], settings, pages=1, mode="daily", client=FakeClient(pages))
    client = FakeClient(pages)
    posts = crawl(None, settings, pages=1, mode="daily", client=client)

    assert client.calls == [("lo裙", 1)]
    assert [post.id for post in posts] == ["w1"]


def test_daily_mode_without_any_keywords_raises(tmp_path):
    settings = make_settings(tmp_path)
    with pytest.raises(ValueError, match="daily 模式"):
        crawl(None, settings, pages=1, mode="daily", client=FakeClient({}))


def test_unknown_mode_is_rejected(tmp_path):
    settings = make_settings(tmp_path)
    with pytest.raises(ValueError, match="未知采集模式"):
        crawl(["Lolita"], settings, pages=1, mode="whatever", client=FakeClient({}))
