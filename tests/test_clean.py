"""清洗规则测试。

这些规则看起来琐碎，但每一条都直接影响检索质量，
所以每一项都有独立用例，回归时能立刻定位是哪条规则坏了。
"""

from __future__ import annotations

from src.preprocessing.clean import clean_text, content_hash, dedupe_posts
from src.schema import Post


def test_removes_url_mention_topic_and_emoji():
    raw = "转发微博 @裙摆与猫：这条裙子太好看了😍 https://t.cn/A6xyzQ #Lolita# 求链接 O网页链接"
    out = clean_text(raw)
    assert "http" not in out
    assert "@" not in out
    assert "#" not in out
    assert "😍" not in out
    assert "转发微博" not in out
    assert "网页链接" not in out
    # 话题文字要保留：它常常是有效检索词
    assert "Lolita" in out


def test_strips_html_and_entities():
    out = clean_text('<a href="https://x.com">正文</a>&nbsp;内容')
    assert "正文" in out
    assert "<a" not in out
    assert "href" not in out


def test_normalizes_fullwidth_and_collapses_whitespace():
    out = clean_text("Ｌｏｌｉｔａ　　是　一种　风格")
    assert "Lolita" in out
    assert "  " not in out
    assert "　" not in out


def test_keeps_chinese_punctuation_intact():
    """中文标点必须保留：这批字符一旦被转成半角，展示检索原文时会很难看。"""
    out = clean_text("裙撑要先买 A 字撑，不要一上来就买暴力撑。真的！你信吗？")
    for mark in "，。！？":
        assert mark in out, f"{mark} 不应被转成半角"
    assert "A" in out


def test_strips_repost_chain_but_keeps_content():
    out = clean_text("// @某人：说得好 // @另一个人：确实 裙撑很重要")
    assert "某人" not in out
    assert "裙撑很重要" in out


def test_empty_input_is_safe():
    assert clean_text("") == ""
    assert clean_text(None) == ""  # type: ignore[arg-type]


def test_content_hash_is_stable_across_calls():
    assert content_hash("同样的文本") == content_hash("同样的文本")
    assert content_hash("文本A") != content_hash("文本B")


def test_dedupe_keeps_the_more_popular_copy():
    low = Post(id="a", text="同一条内容", likes=1)
    high = Post(id="b", text="同一条内容", likes=99)
    result = dedupe_posts([low, high])
    assert len(result) == 1
    assert result[0].id == "b"
