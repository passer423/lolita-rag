"""热门推荐接口的测试。

重点不在「能返回 JSON」，而在两件容易做错的事：

1. **没有语料时必须说清为什么空**——否则用户以为功能坏了（这里注入一个空目录）；
2. **建议查询不能是语料里不存在的词**——那种查询会被检索闸门直接拒答，
   点「用这个词检索」得到空结果，比没有按钮更糟。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.config import Settings
from src.pipeline.hot_posts import suggest_query
from src.schema import Post
from src.web.app import create_app


class FakeRetriever:
    def __init__(self, frequencies: dict[str, int] | None = None):
        self.bm25 = SimpleNamespace(
            document_frequency=lambda token: (frequencies or {}).get(token, 0)
        )

    def search(self, question, top_k=None, recall_k=None):
        return []


def write_corpus(tmp_path, posts: list[dict]) -> None:
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "posts.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in posts) + "\n", encoding="utf-8"
    )


def make_client(tmp_path, posts: list[dict], frequencies: dict[str, int] | None = None):
    write_corpus(tmp_path, posts)
    bundle = SimpleNamespace(retriever=FakeRetriever(frequencies), chunks=[], backend="simple")
    settings = Settings(data_dir=tmp_path, llm_api_key="", hot_limit=5)
    return TestClient(create_app(bundle=bundle, settings=settings))


def corpus_row(pid: str, *, text: str, when: str, author: str = "某人", likes=0, comments=0, reposts=0):
    return {
        "id": pid,
        "text": text,
        "author": author,
        "created_at": when,
        "url": f"https://m.weibo.cn/detail/{pid}",
        "likes": likes,
        "comments": comments,
        "reposts": reposts,
        "source": "weibo",
    }


# ------------------------------------------------------------------ 接口

def test_hot_endpoint_returns_items_with_anchor_and_window(tmp_path):
    client = make_client(
        tmp_path,
        [
            corpus_row("a", text="lolita 柄图记录：今天的 jsk 很好看", when="2026-10-12 20:00", likes=5),
            corpus_row("b", text="lolita 图透 转发抽奖 专平一位", when="2026-10-11 10:00", likes=99, author="品牌"),
        ],
    )
    with client:
        data = client.get("/api/hot", params={"days": 1}).json()

    assert data["anchor_label"] == "2026-10-12"
    assert data["window_days"] == 1
    assert data["total_posts"] == 2
    assert [item["id"] for item in data["items"]] == ["a"], "窗口只含最新一天"
    assert data["items"][0]["likes"] == 5


def test_hot_endpoint_widens_with_days_parameter(tmp_path):
    client = make_client(
        tmp_path,
        [
            corpus_row("a", text="lolita 柄图记录：今天的 jsk 很好看", when="2026-10-12 20:00", likes=1),
            corpus_row("b", text="lolita 老帖图透 转发抽奖", when="2026-10-09 10:00", likes=99, author="品牌"),
        ],
    )
    with client:
        one = client.get("/api/hot", params={"days": 1}).json()
        seven = client.get("/api/hot", params={"days": 7}).json()

    assert [item["id"] for item in one["items"]] == ["a"]
    assert [item["id"] for item in seven["items"]] == ["b", "a"]


def test_hot_endpoint_says_why_it_is_empty_when_corpus_missing(tmp_path):
    """CSV 语料缺失时不能只给个空数组——那看起来就像功能坏了。"""
    bundle = SimpleNamespace(retriever=FakeRetriever(), chunks=[], backend="simple")
    client = TestClient(create_app(bundle=bundle, settings=Settings(data_dir=tmp_path, llm_api_key="")))
    with client:
        data = client.get("/api/hot").json()

    assert data["items"] == []
    assert "读不到博文语料" in data["hint"]


def test_hot_endpoint_explains_an_empty_day(tmp_path):
    client = make_client(
        tmp_path,
        [corpus_row("x", text="今天天气不错，我去吃了一顿火锅，很开心", when="2026-10-12 20:00", likes=50)],
    )
    with client:
        data = client.get("/api/hot").json()

    assert data["items"] == []
    assert "没有同时满足" in data["hint"]
    assert data["skipped"]["domain"] == 1, "筛选统计要一起返回，用户才知道是为什么空的"


def test_hot_endpoint_suggests_a_query_that_exists_in_the_corpus(tmp_path):
    """建议查询必须是语料里真实存在的词：df=0 的词会被检索闸门拒答。"""
    client = make_client(
        tmp_path,
        [
            corpus_row(
                "a",
                text="lolita 柄图记录：蓟草之冠 的 jsk 很好看",
                when="2026-10-12 20:00",
                likes=5,
            )
        ],
        # 只有「柄图」在语料里存在；「蓟草之冠」df=0（会被闸门拒答，不能推荐）
        frequencies={"柄图": 12},
    )
    with client:
        data = client.get("/api/hot").json()

    assert data["items"][0]["suggested_query"] == "柄图"


def test_hot_endpoint_validates_parameters(tmp_path):
    client = make_client(tmp_path, [corpus_row("a", text="lolita 图透 记录", when="2026-10-12 20:00")])
    with client:
        assert client.get("/api/hot", params={"days": 0}).status_code == 422
        assert client.get("/api/hot", params={"limit": 999}).status_code == 422


# ------------------------------------------------------------------ 建议查询

def test_suggest_query_prefers_a_domain_term_over_a_rarer_generic_word():
    """实测教训：按「最罕见的内容词」挑会挑出「800」「plq」「穿刚」这类噪声。

    罕见 ≠ 有主题性。所以领域词优先，哪怕它不是最罕见的。
    """
    frequencies = {"柄图": 40, "jsk": 180, "800": 1, "plq": 1}
    query = suggest_query("jsk 与柄图 记录，收藏过800 见plq置顶", lambda t: frequencies.get(t, 0))
    assert query in {"柄图", "jsk"}, f"应当从领域词里挑，实际挑到 {query!r}"


def test_suggest_query_skips_numbers_and_short_latin_noise_in_fallback():
    """没有领域词时也不能挑出「800」「mm」「plq」这种当查询。"""
    frequencies = {"800": 1, "mm": 1, "葡萄园": 30}
    query = suggest_query("漂亮mm穿的葡萄园 收藏过800", lambda t: frequencies.get(t, 0))
    assert query == "葡萄园"


def test_suggest_query_skips_terms_absent_from_the_corpus():
    """df=0 的词当查询会被闸门拒答，点了按钮得到空结果 —— 比没有按钮更糟。

    所有候选词都不在语料里时，退化为正文片段（至少点了有事发生）。
    """
    text = "蓟草之冠 绝版裙子"
    assert suggest_query(text, lambda t: 0) == text


def test_suggest_query_skips_stopwords_and_question_words():
    assert suggest_query("这条裙子怎么洗呢 柄图", lambda t: 3) == "柄图"


def test_suggest_query_falls_back_to_a_snippet_without_statistics():
    text = "黑雪松 jsk 就绪，人台与细节展示"
    query = suggest_query(text, None)
    assert query and query in text
