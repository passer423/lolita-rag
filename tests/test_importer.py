"""手工导入路径的测试。

这条路径的意义：微博关键词检索必须要登录态，如果不想让 Cookie 进聊天记录，
用户可以自己在浏览器里保存响应/页面再交给导入器。所以解析的健壮性很重要——
文件格式千奇百怪，**解析失败也绝不能抛异常**，只能把「看到了什么」记下来。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.crawler.importer import _extract_balanced, import_paths, load_file
from src.schema import Post

FIXTURES = Path(__file__).parent / "fixtures"


def _write_har(tmp_path, entries: list[dict]) -> Path:
    path = tmp_path / "capture.har"
    path.write_text(json.dumps({"log": {"entries": entries}}, ensure_ascii=False), encoding="utf-8")
    return path


def _html_entry(url: str, body: str) -> dict:
    return {"request": {"url": url}, "response": {"content": {"mimeType": "text/html; charset=UTF-8", "text": body}}}


def _json_entry(url: str, body: str) -> dict:
    return {"request": {"url": url}, "response": {"content": {"mimeType": "application/json", "text": body}}}


def test_imports_har_with_many_pages_at_once(tmp_path):
    """HAR 是扩充语料最省事的路子：一次导出就包含所有翻过的页面。"""
    pc_html = (FIXTURES / "saved_pc_search_page.html").read_text(encoding="utf-8")
    container = (FIXTURES / "weibo_container_page.json").read_text(encoding="utf-8")
    path = _write_har(
        tmp_path,
        [
            _html_entry("https://s.weibo.com/weibo?q=lolita&page=1", pc_html),
            _html_entry("https://s.weibo.com/weibo?q=lolita&page=2", pc_html),
            _json_entry("https://m.weibo.cn/api/container/getIndex?containerid=x", container),
        ],
    )

    item = load_file(path)
    assert item.kind == "har"
    assert item.posts == 8, "两页 PC 页面各 2 条 + container 4 条"
    assert any("HAR" in note and "pc-search" in note for note in item.notes)


def test_har_ignores_non_weibo_and_binary_entries(tmp_path):
    path = _write_har(
        tmp_path,
        [
            _html_entry("https://example.com/page", "<html>无关站点</html>"),
            {
                "request": {"url": "https://s.weibo.com/img/a.png"},
                "response": {"content": {"mimeType": "image/png", "encoding": "base64", "text": "AAAA"}},
            },
        ],
    )
    item = load_file(path)
    assert item.posts == 0
    assert any("没有可用的微博响应" in note for note in item.notes)


def test_har_without_entries_is_reported(tmp_path):
    path = tmp_path / "broken.har"
    path.write_text(json.dumps({"log": {}}), encoding="utf-8")
    item = load_file(path)
    assert item.posts == 0
    assert any("log.entries" in note for note in item.notes)


def test_import_dedupes_overlapping_pages_from_har(tmp_path):
    """翻页时相邻页常有重叠，导入必须去重，否则重复内容会污染检索。"""
    pc_html = (FIXTURES / "saved_pc_search_page.html").read_text(encoding="utf-8")
    path = _write_har(
        tmp_path,
        [
            _html_entry("https://s.weibo.com/weibo?q=lolita&page=1", pc_html),
            _html_entry("https://s.weibo.com/weibo?q=lolita&page=2", pc_html),
        ],
    )
    result = import_paths([path])
    assert len(result.posts) == 2, "两页内容相同，去重后应只剩 2 条"
    assert any("去重" in note for note in result.notes)


def test_exclude_filter_drops_promo_posts():
    """营销贴会稀释检索结果，可按关键词排除。"""
    result = import_paths([FIXTURES / "saved_pc_search_page.html"], exclude=["暴力撑"])

    assert len(result.posts) == 1
    assert all("暴力撑" not in post.text for post in result.posts)
    assert any("按关键词排除" in note for note in result.notes)


def test_exclude_with_no_match_keeps_everything():
    result = import_paths([FIXTURES / "saved_pc_search_page.html"], exclude=["不存在的词"])
    assert len(result.posts) == 2
    assert not any("排除" in note for note in result.notes)


def test_imports_pc_search_page_saved_by_ctrl_s():
    """Ctrl+S 保存的 s.weibo.com 搜索页，结构与移动端 API 完全不同。"""
    item = load_file(FIXTURES / "saved_pc_search_page.html")
    assert item.kind == "html/pc-search"
    assert item.posts == 2, "没有 mid 的广告卡与用户卡都必须跳过"
    assert {post.id for post in item.parsed} == {"5352896195725528", "5352896195725599"}

    first = next(post for post in item.parsed if post.id.endswith("528"))
    assert first.author == "示例作者甲"
    assert first.created_at == "10月11日 18:09"
    assert first.url == "https://m.weibo.cn/detail/5352896195725528"
    assert "黑蔷薇" in first.text
    assert "<i" not in first.text and "wbicon" not in first.text


def test_pc_search_page_prefers_the_full_text_node():
    """长文时 feed_list_content 是被截断的，必须优先取 feed_list_content_full。"""
    item = load_file(FIXTURES / "saved_pc_search_page.html")
    second = next(post for post in item.parsed if post.id.endswith("599"))
    assert "展开" not in second.text
    assert "暴力撑" in second.text and "阴干" in second.text


def test_pc_search_page_strips_fold_controls_and_icon_glyphs():
    """「收起」「展开」控件与其图标字体字形（d/c/L）会混进正文——必须按属性精确剔除。

    真实页面里长这样：
        <a action-type="fl_fold">收起<i class="wbicon">d</i></a>
    实测未处理时正文会以「… lolita上新 收起 d」结尾。
    """
    item = load_file(FIXTURES / "saved_pc_search_page.html")
    second = next(post for post in item.parsed if post.id.endswith("599"))
    assert "收起" not in second.text
    assert " d" not in second.text and not second.text.endswith("d")
    assert second.text.endswith("就够了。"), f"正文结尾不应残留控件文字：{second.text[-20:]!r}"


def test_pc_search_page_without_post_cards_reports_what_it_saw(tmp_path):
    path = tmp_path / "empty.html"
    path.write_text(
        '<html><body><div class="card-wrap"><div class="card">用户卡片</div></div></body></html>',
        encoding="utf-8",
    )
    item = load_file(path)
    assert item.posts == 0
    assert any("feed_list_item" in note for note in item.notes), "要说明看到了什么，便于补适配"


def test_imports_container_api_response():
    item = load_file(FIXTURES / "weibo_container_page.json")
    assert item.kind == "json"
    assert item.posts == 4
    assert {post.id for post in item.parsed} == {"9001", "5001", "5002", "5003"}


def test_imports_page_with_embedded_render_data():
    """页面内嵌 JSON 里有干扰字符 ] } " ，括号配平必须能扛住。"""
    item = load_file(FIXTURES / "saved_search_page.html")
    assert item.kind == "html/$render_data"
    assert item.posts == 2
    assert {post.id for post in item.parsed} == {"7001", "7002"}

    first = next(post for post in item.parsed if post.id == "7001")
    assert "JSK" in first.text
    assert "<b>" not in first.text
    assert "🍓" not in first.text          # &#127827; 解出来是 emoji，应被清掉
    assert first.author == "页面导入测试"
    assert first.likes == 12


def test_imports_own_jsonl_format(tmp_path):
    path = tmp_path / "posts.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in [
                {"id": "8001", "text": "第一条 jsonl 记录，讲的是柄图。", "author": "甲", "likes": 3},
                {"id": "8002", "text": "第二条 jsonl 记录，讲的是裙撑。", "author": "乙"},
            ]
        ),
        encoding="utf-8",
    )
    item = load_file(path)
    assert item.kind == "posts.jsonl"
    assert item.posts == 2


def test_unknown_structure_reports_instead_of_raising(tmp_path):
    path = tmp_path / "weird.json"
    path.write_text(json.dumps({"hello": "world"}, ensure_ascii=False), encoding="utf-8")

    item = load_file(path)
    assert item.posts == 0
    assert any("未识别的对象结构" in note for note in item.notes)
    assert item.parsed == []


def test_broken_json_is_reported_not_raised(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{这不是合法 JSON", encoding="utf-8")

    item = load_file(path)
    assert item.posts == 0
    assert any("JSON 解析失败" in note for note in item.notes)


def test_login_page_html_reports_what_it_saw(tmp_path):
    """识别不了时必须留下说明，而不是静默返回空。

    断言刻意不写死具体措辞：HTML 有两条解析路径（$render_data / PC 搜索页），
    哪条都不匹配时只要说清「看到了什么」就够了。
    """
    path = tmp_path / "login.html"
    path.write_text("<html><head><title>Sina Visitor System</title></head><body></body></html>", encoding="utf-8")

    item = load_file(path)
    assert item.posts == 0
    assert item.notes, "不能静默返回空"
    assert any("feed_list_item" in note or "$render_data" in note for note in item.notes)


def test_directory_scan_and_cross_file_dedupe(tmp_path):
    """同一份响应存了两次（例如两页有重叠），去重后不应重复入库。"""
    payload = {
        "ok": 1,
        "data": {
            "cards": [
                {
                    "card_type": 9,
                    "mblog": {
                        "id": "9001",
                        "text": "重复出现的同一条博文，讲的是柄图与茶会。",
                        "user": {"screen_name": "甲"},
                    },
                }
            ]
        },
    }
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    result = import_paths([tmp_path])
    assert len(result.files) == 2
    assert len(result.posts) == 1, "内容相同的博文应当去重"
    assert any("去重" in note for note in result.notes)


def test_missing_path_is_reported(tmp_path):
    result = import_paths([tmp_path / "不存在.json"])
    assert result.posts == []
    assert any("路径不存在" in note for note in result.notes)


def test_extract_balanced_ignores_brackets_inside_strings():
    text = 'prefix ["a]b", {"k": "}"}, "c"] suffix'
    assert _extract_balanced(text, text.index("[")) == '["a]b", {"k": "}"}, "c"]'


def test_import_paths_cleans_and_counts(tmp_path):
    path = tmp_path / "one.json"
    path.write_text(
        json.dumps(
            {
                "cards": [
                    {
                        "card_type": 9,
                        "mblog": {
                            "id": "9100",
                            "text": "转发微博 @某人：带亮片的柄图别机洗😊 https://t.cn/xyz",
                            "user": {"screen_name": "甲"},
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = import_paths([path])
    assert len(result.posts) == 1
    text = result.posts[0].text
    assert "http" not in text and "@" not in text and "😊" not in text
    assert "亮片" in text and "柄图" in text


def test_imported_posts_are_mergeable(tmp_path, monkeypatch):
    """导入结果必须能直接并进 data/raw/posts.jsonl，供 build_index 使用。"""
    from src.config import Settings
    from src.pipeline.crawl import merge_and_save
    from src.storage import load_posts

    settings = Settings(data_dir=tmp_path)
    result = import_paths([FIXTURES / "weibo_container_page.json"])
    assert result.posts

    before, after = merge_and_save(result.posts, settings)
    assert before == 0 and after == len(result.posts)

    # 再导入一次：应当幂等，不产生重复
    before2, after2 = merge_and_save(result.posts, settings)
    assert before2 == after2 == after
    assert len(load_posts(settings.raw_posts_path)) == after


def test_post_from_dict_requires_id_and_text():
    """jsonl 行缺 id 时不能抛异常——直接跳过。"""
    with pytest.raises(TypeError):
        Post.from_dict({"text": "没有 id"})
