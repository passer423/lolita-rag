"""微博响应解析测试（离线，不联网）。

**这些测试的边界要说清楚**：夹具是手工构造的合成数据，只能证明解析器
对「已知结构 + 边界情况」的处理正确，不能证明真实接口就是这个形状。
真实形状必须在拿到 Cookie 后用 ``scripts/crawl.py --dry-run`` 验证一次。
它仍然有价值：接口结构没变时，任何解析回归都会被立刻发现。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.crawler.weibo_client import WeiboClient

FIXTURE = Path(__file__).parent / "fixtures" / "weibo_container_page.json"


@pytest.fixture()
def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def by_id(payload: dict, post_id: str):
    """按 id 取解析结果——比按下标稳，加夹具不会再连带改一堆断言。"""
    for post in WeiboClient.parse_container(payload):
        if post.id == post_id:
            return post
    raise AssertionError(f"没有解析出 id={post_id} 的博文")


def test_parses_only_valid_mblogs(payload: dict):
    posts = WeiboClient.parse_container(payload)
    # 分组里的 card_type=7、纯空白正文、缺 id 的，都该被丢掉；
    # 而包在 card_type=11 包装卡片里的 9001 必须留下（真实接口第 2 页起就是这个形状）
    assert [post.id for post in posts] == ["9001", "5001", "5002", "5003"]


def test_parses_mblogs_nested_in_wrapper_cards(payload: dict):
    """回归：搜索接口第 2 页起顶层是 card_type=11，博文在 card_group 里。

    旧实现只认顶层 card_type=9，于是第 2 页起整页判空、被当成「已到末页」，
    采集永远只拿到第 1 页（实测 8 个关键词各 13 条，而接口 total=751）。
    """
    posts = WeiboClient.parse_container(payload)
    assert "9001" in [post.id for post in posts]


def test_nested_wrapper_posts_are_deduplicated():
    """同一条博文同时出现在包装卡片和分组里时，只能算一条。"""
    mblog = {"id": "777", "text": "重复出现的博文", "user": {"screen_name": "重复"}}
    payload = {
        "data": {
            "cards": [
                {"card_type": 11, "card_group": [{"card_type": 9, "mblog": mblog}]},
                {"card_type": 9, "mblog": dict(mblog)},
            ]
        }
    }
    assert [post.id for post in WeiboClient.parse_container(payload)] == ["777"]


def test_accepts_string_card_type():
    """card_type 实测既有整数 9 也有字符串 "9"。"""
    payload = {
        "data": {
            "cards": [
                {"card_type": "11", "card_group": [
                    {"card_type": "9", "mblog": {"id": "888", "text": "字符串 card_type", "user": {}}}
                ]}
            ]
        }
    }
    assert [post.id for post in WeiboClient.parse_container(payload)] == ["888"]


def test_extracts_metadata_and_builds_detail_url(payload: dict):
    first = by_id(payload, "5001")
    assert first.author == "裙摆与猫"
    assert first.url == "https://m.weibo.cn/detail/5001"
    assert first.source == "weibo"
    assert (first.likes, first.comments, first.reposts) == (328, 76, 12)
    assert first.created_at.startswith("Mon Mar 02")


def test_missing_counts_default_to_zero(payload: dict):
    second = by_id(payload, "5002")
    assert second.likes == 1204        # 整数 id 也被正确转成了字符串
    assert second.comments == 0
    assert second.reposts == 0


def test_cleans_html_url_topic_emoji_and_entities(payload: dict):
    first = by_id(payload, "5001")
    assert "<b>" not in first.text
    assert "http" not in first.text
    assert "#" not in first.text
    assert "🍓" not in first.text
    assert "Lolita" in first.text        # 话题文字保留
    assert "JSK" in first.text

    second = by_id(payload, "5002")
    assert "\xa0" not in second.text      # &nbsp; 要变成普通空格
    assert "😊" not in second.text
    # 中文标点不能被改成半角
    assert "：" in second.text


def test_repost_flag_follows_retweeted_status(payload: dict):
    assert by_id(payload, "5001").is_repost is False
    assert by_id(payload, "5003").is_repost is True


def test_tolerates_malformed_payload():
    assert WeiboClient.parse_container({}) == []
    assert WeiboClient.parse_container({"ok": 0}) == []
    assert WeiboClient.parse_container({"data": {"cards": None}}) == []
    assert WeiboClient.parse_container({"data": {"cards": [{"card_type": 9}]}}) == []


def test_requires_cookie_before_any_request():
    from src.config import Settings
    from src.crawler.weibo_client import WeiboCrawlError

    with pytest.raises(WeiboCrawlError, match="WEIBO_COOKIE"):
        WeiboClient(Settings(weibo_cookie=""))


# --------------------------------------------------- 采尽页与真故障的区分（真实回归）
class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return _FakeResponse(self.payload)


def _client_for(payload: dict) -> WeiboClient:
    from src.config import Settings

    return WeiboClient(
        Settings(weibo_cookie="SUB=x", crawl_interval=0, crawl_max_retry=1),
        session=_FakeSession(payload),
    )


def test_exhausted_search_page_is_treated_as_no_results_not_an_error():
    """实测：关键词采尽时微博返回 ``ok=0`` + 「这里还没有内容」。

    旧实现把它当异常抛出，后果是**整轮采集中断**：16 个知识型关键词只采到 5 个，
    后面的一个都没跑到。长尾关键词往往一两页就见底，所以这个「假故障」影响很大。
    """
    client = _client_for({"ok": 0, "msg": "这里还没有内容", "data": {"cards": []}})
    assert client.search_page("某个长尾关键词", 3) == []


def test_other_ok_zero_payloads_still_raise():
    """对照：真正的异常（风控 / 接口变化）不能被当成空页吞掉。"""
    from src.crawler.weibo_client import WeiboCrawlError

    client = _client_for({"ok": 0, "msg": "您的账号存在异常行为"})
    with pytest.raises(WeiboCrawlError, match="接口返回异常"):
        client.search_page("Lolita", 1)
