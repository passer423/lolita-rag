"""长文补全的测试。

背景是一个**我自己先判错、再查出来**的问题：我以为模型出现了「资料里有却说没有」的
语义不忠实，追下去发现是那条博文在语料里只有 141 字（真实 1365 字）——**被微博接口截断了**，
模型是对的、我错了。所以这一组测试要守住三件事：

1. 截断能被**认出来**（各种标记写法）；
2. 取回全文就**替换**，并且不改动别的字段；
3. 取不回来**不能假装完整**——必须留下显式的截断说明。
"""

from __future__ import annotations

from src.pipeline.longtext import (
    TRUNCATION_NOTE,
    candidates_in,
    is_truncated,
    refresh_long_posts,
    strip_truncation_marker,
    summarize,
)
from src.schema import Post


def post(pid: str, text: str, **extra) -> Post:
    payload = {"id": pid, "text": text, "author": "某人", "source": "weibo", **extra}
    return Post.from_dict(payload)


# ------------------------------------------------------------------ 识别

def test_detects_the_marker_variants_weibo_actually_emits():
    assert is_truncated("正文写到一半 ... 全文")
    assert is_truncated("正文写到一半…全文")
    assert is_truncated("正文写到一半 全文")
    assert is_truncated("正文写到一半...全文   ")
    assert is_truncated("LOLITA 记录 关于洗涤标志和lo裙保养 ... 全文")


def test_complete_posts_are_not_flagged():
    assert not is_truncated("裙子：藤上春 Time：10.20 Price：jsk=295")
    assert not is_truncated("全文都在这里了")          # 「全文」出现在句中不算
    assert not is_truncated("")


def test_marker_is_replaced_by_an_explicit_note_not_silently_deleted():
    """删掉标记会让截断**不可见**：博文看起来完整，实际少了 90%。"""
    text = strip_truncation_marker("博文写到一半 ... 全文")
    assert TRUNCATION_NOTE in text
    assert not is_truncated(text)
    assert "博文写到一半" in text


# ------------------------------------------------------------------ 补全

def test_successful_fetch_replaces_text_and_keeps_everything_else():
    original = post("p1", "安东尼蛋糕再贩 ... 全文", likes=5, url="https://e/p1")
    full = "安东尼蛋糕再贩：本次我们将提升各方面的效能与细节，具体预约/尾款详情于10.3-10.9左右公布"

    refreshed, stats = refresh_long_posts([original], lambda pid: full)

    assert stats.fetched == 1 and stats.failed == 0
    assert refreshed[0].text == full
    # 其余字段一个都不能丢
    assert refreshed[0].id == "p1"
    assert refreshed[0].likes == 5
    assert refreshed[0].url == "https://e/p1"
    assert refreshed[0].author == "某人"
    assert stats.examples and "141" not in stats.examples[0]  # 示例里会写长度变化


def test_failed_fetch_marks_truncation_instead_of_pretending_it_is_complete():
    original = post("p2", "博文写到一半 ... 全文")
    refreshed, stats = refresh_long_posts([original], lambda pid: "")

    assert stats.failed == 1 and stats.fetched == 0
    assert TRUNCATION_NOTE in refreshed[0].text
    assert not is_truncated(refreshed[0].text)


def test_fetch_exception_is_counted_not_raised():
    """单条失败不能中断整批补全——331 条跑一半崩掉会很难受。"""

    def boom(pid: str) -> str:
        raise RuntimeError("连接超时")

    refreshed, stats = refresh_long_posts([post("p3", "写到一半 ... 全文")], boom)
    assert stats.failed == 1
    assert TRUNCATION_NOTE in refreshed[0].text


def test_complete_posts_are_left_alone_and_not_requested():
    calls: list[str] = []

    def fetch(pid: str) -> str:
        calls.append(pid)
        return "should not be called"

    posts = [post("ok", "这条是完整的正文，不需要补。"), post("bad", "这条被截断 ... 全文")]
    refreshed, stats = refresh_long_posts(posts, fetch)

    assert calls == ["bad"], "完整的博文不该产生请求"
    assert stats.candidates == 1
    assert refreshed[0].text == "这条是完整的正文，不需要补。"


def test_limit_stops_requesting_and_counts_the_rest_as_skipped():
    posts = [post(f"p{i}", f"第{i}条被截断 ... 全文") for i in range(5)]
    refreshed, stats = refresh_long_posts(
        posts, lambda pid: "补回来的完整正文，内容足够长，不再带截断标记。", limit=2
    )

    assert stats.fetched == 2
    assert stats.skipped == 3
    assert len(refreshed) == 5
    assert is_truncated(refreshed[4].text), "跳过的保持原样（下次再补）"


def test_fetched_text_that_still_looks_truncated_is_rejected():
    """接口如果又返回截断版（或以「全文」结尾），不能当成功——否则截断会被标成已修复。

    这个守卫是刻意的，代价是「正文恰好以『全文』二字结尾」的极少数帖子会被误判为失败，
    那也比**悄悄保留截断**强。
    """
    refreshed, stats = refresh_long_posts(
        [post("p9", "写到一半 ... 全文")], lambda pid: "依然被截断 ... 全文"
    )
    assert stats.failed == 1 and stats.fetched == 0
    assert TRUNCATION_NOTE in refreshed[0].text


def test_refresh_does_not_mutate_the_input_objects():
    """返回新对象而不是就地改：批处理中途失败时原始数据仍可用。"""
    original = post("p4", "写到一半 ... 全文")
    refresh_long_posts([original], lambda pid: "全文")
    assert is_truncated(original.text)


def test_empty_fetch_result_is_treated_as_failure():
    """接口返回 ok=1 但 longTextContent 为空 → 视为失败，不能把正文清空。"""
    refreshed, stats = refresh_long_posts([post("p5", "写到一半 ... 全文")], lambda pid: "   ")
    assert stats.failed == 1
    assert "写到一半" in refreshed[0].text


def test_candidates_and_summary():
    posts = [post("a", "完整"), post("b", "截断 ... 全文"), post("c", "也截断 全文")]
    assert [item.id for item in candidates_in(posts)] == ["b", "c"]

    _, stats = refresh_long_posts(posts, lambda pid: "")
    text = summarize(stats)
    assert "待补全 2 条" in text and "失败 2" in text
