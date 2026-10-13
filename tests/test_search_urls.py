"""搜索链接生成的测试。

链接是给人点的，写错了就白费一次人工操作，所以值得单独测。
"""

from __future__ import annotations

from urllib.parse import unquote

from src.crawler.search_urls import DEFAULT_KEYWORDS, PROMO_EXCLUDE, build_search_urls


def test_builds_page_range_per_keyword():
    urls = build_search_urls(["lo裙", "JSK"], pages=3)
    assert len(urls) == 6
    assert "page=1" in urls[0] and "page=3" in urls[2]
    assert "page=1" in urls[3] and "page=3" in urls[5]


def test_url_encodes_chinese_keywords():
    url = build_search_urls(["洛丽塔 穿搭"], pages=1)[0]
    assert " " not in url, "空格必须被编码，否则链接会被截断"
    assert "%E6%B4%9B%E4%B8%BD%E5%A1%94" in url.upper()
    assert unquote(url.split("q=")[1].split("&")[0]) == "洛丽塔 穿搭"


def test_blank_keywords_are_skipped():
    assert build_search_urls(["", "  ", "lo裙"], pages=1) == [
        "https://s.weibo.com/weibo?q=lo%E8%A3%99&page=1"
    ]


def test_pages_at_least_one():
    assert len(build_search_urls(["lo裙"], pages=0)) == 1


def test_default_keywords_are_domain_specific():
    """默认词表要偏领域词（超话名/黑话），不能只有一个泛词。"""
    assert len(DEFAULT_KEYWORDS) >= 5
    assert any("超话" in word for word in DEFAULT_KEYWORDS)
