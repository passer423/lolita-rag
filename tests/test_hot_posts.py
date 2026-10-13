"""热门推荐的测试。

这个功能的两个关键词都容易变成骗人的东西，所以测试重点不在「能不能排序」，而在：

1. **「当日」的锚点必须是语料里的最新时间，不是墙上时钟**——实测墙上时钟那天语料里
   一条都没有，若按 now 算，这一栏会长期空白（或者更糟：拿旧帖冒充今日）；
2. **相关性过滤要不区分大小写**——词表写 `lolita`、博文写 `LOLITA` 时不能误杀；
3. **域外热帖要被挡住**——实测「最近 3 天按点赞排」第一名是 11601 赞的明星时装周帖；
4. 同作者限量、过短正文过滤、空语料不炸。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.pipeline.hot_posts import (
    TZ,
    clean_display_text,
    hot_score,
    is_domain_related,
    load_domain_terms,
    parse_post_time,
    select_hot_posts,
)
from src.schema import Post


def post(pid: str, *, when="2026-10-12 20:00", text="一条 lolita 相关的小裙子记录", author="某人", likes=0, comments=0, reposts=0) -> Post:
    return Post(
        id=pid,
        text=text,
        author=author,
        created_at=when,
        url=f"https://m.weibo.cn/detail/{pid}",
        likes=likes,
        comments=comments,
        reposts=reposts,
        source="weibo",
    )


# ------------------------------------------------------------------ 时间解析

def test_parses_the_three_timestamp_formats_in_the_corpus():
    """三条采集路径三种格式，都要能解析——否则热门栏里会凭空少掉一批帖子。"""
    weibo = parse_post_time("Fri Oct 09 17:45:00 +0800 2026")
    assert weibo == datetime(2026, 10, 9, 17, 45, tzinfo=TZ)

    iso = parse_post_time("2026-10-12 20:26:40")
    assert iso == datetime(2026, 10, 12, 20, 26, 40, tzinfo=TZ)
    assert parse_post_time("2026-10-12") == datetime(2026, 10, 12, tzinfo=TZ)

    # HTML 导入路径缺年份，必须由 default_year 补；不传就不猜
    assert parse_post_time("10月11日 20:45") is None
    assert parse_post_time("10月11日 20:45", default_year=2026) == datetime(
        2026, 10, 11, 20, 45, tzinfo=TZ
    )


def test_unparseable_time_returns_none_instead_of_guessing():
    for bad in ("", "   ", "刚刚", "3小时前", None):
        assert parse_post_time(bad) is None


# ------------------------------------------------------------------ 相关性

def test_domain_match_is_case_insensitive():
    """实测踩过：词表是 lolita、博文是 LOLITA，第一版把这个真·圈内帖判成域外。"""
    terms = ("lolita", "jsk", "柄图")
    assert is_domain_related("LOLITA 今天穿的是黑蔷薇", terms)
    assert is_domain_related("这条 JSK 的柄图很好看", terms)
    assert not is_domain_related("闪现魔都 hello时装周", terms)


def test_domain_terms_load_from_the_shipped_dict():
    terms = load_domain_terms()
    assert "柄图" in terms and "裙撑" in terms
    assert all(term == term.lower() for term in terms), "统一小写，匹配时才不会漏"


# ------------------------------------------------------------------ 打分

def test_hot_score_weights_reposts_and_comments_above_likes():
    plain = post("a", likes=10)
    discussed = post("b", likes=0, comments=5)      # 0 + 2*5 = 10
    shared = post("c", likes=0, reposts=4)          # 0 + 3*4 = 12
    assert hot_score(plain) == 10
    assert hot_score(discussed) == 10
    assert hot_score(shared) == 12


# ------------------------------------------------------------------ 选帖

def test_anchor_is_the_corpus_latest_time_not_wall_clock():
    """核心行为：今天（墙上时钟）没有帖子时，锚点落到语料最新那天，而不是空栏。"""
    posts = [
        post("old", when="2020-01-01 10:00", likes=99999),
        post("new1", when="2026-10-12 20:00", likes=10),
        post("new2", when="2026-10-12 19:00", likes=5),
    ]
    selection = select_hot_posts(posts, window_days=1)

    assert selection.anchor == datetime(2026, 10, 12, 20, 0, tzinfo=TZ)
    assert selection.anchor_label == "2026-10-12"
    assert [item.post.id for item in selection.items] == ["new1", "new2"]
    assert all(item.post.id != "old" for item in selection.items), "旧帖不能冒充当日"


def test_window_days_widens_the_candidate_pool():
    # 作者各不相同：同作者超过 per_author_cap 会被限量规则挡掉，那是另一条用例的事
    posts = [
        post("d0", when="2026-10-12 20:00", likes=10, author="A"),
        post("d2", when="2026-10-10 20:00", likes=50, author="B"),
        post("d5", when="2026-10-07 20:00", likes=99, author="C"),
    ]
    one_day = select_hot_posts(posts, window_days=1)
    three_days = select_hot_posts(posts, window_days=3)
    week = select_hot_posts(posts, window_days=7)

    assert [item.post.id for item in one_day.items] == ["d0"]
    assert [item.post.id for item in three_days.items] == ["d2", "d0"]
    assert [item.post.id for item in week.items] == ["d5", "d2", "d0"]


def test_off_topic_viral_post_is_filtered_out():
    """实测：最近 3 天按点赞排第一名是 11601 赞的明星时装周帖，与 Lolita 无关。"""
    posts = [
        post("star", when="2026-10-12 12:00", text="闪现魔都 hello时装周 记璐美好", likes=11601, comments=2652, reposts=2592),
        post("lo", when="2026-10-12 11:00", text="蓟草之冠 背心 jsk 段段 sk 记录", likes=312, comments=200, reposts=934),
    ]
    selection = select_hot_posts(posts, window_days=3)

    assert [item.post.id for item in selection.items] == ["lo"]
    assert selection.skipped["domain"] == 1


def test_requiring_domain_can_be_turned_off():
    posts = [post("star", when="2026-10-12 12:00", text="闪现魔都 hello时装周", likes=100)]
    assert select_hot_posts(posts, require_domain=True).items == []
    assert len(select_hot_posts(posts, require_domain=False).items) == 1


def test_short_or_non_chinese_posts_are_skipped():
    posts = [
        post("tiny", when="2026-10-12 12:00", text="OOTD", likes=100),
        post("emoji", when="2026-10-12 11:00", text="ദ്ദി˶>ᴗo)", likes=90),
        post("good", when="2026-10-12 10:00", text="lolita 柄图记录：这条 jsk 很好看", likes=1),
    ]
    selection = select_hot_posts(posts, window_days=1)

    assert [item.post.id for item in selection.items] == ["good"]
    assert selection.skipped["too_short"] == 2


def test_one_author_cannot_flood_the_list():
    """实测一个品牌的转发抽奖能占满整栏，所以要限每位作者的最大条数。"""
    posts = [
        post(f"brand{i}", when=f"2026-10-12 1{i}:00", text="lolita 新品图透 转发抽奖 专平一位", author="绵羊泡芙", likes=100 - i)
        for i in range(4)
    ] + [post("other", when="2026-10-12 09:00", text="lolita 日常记录 今天穿的 jsk", author="另一个人", likes=1)]
    selection = select_hot_posts(posts, window_days=1, limit=5, per_author_cap=2)

    authors = [item.post.author for item in selection.items]
    assert authors.count("绵羊泡芙") == 2
    assert "另一个人" in authors
    assert selection.skipped["duplicate_author"] == 2


def test_skipped_counts_make_the_filtering_transparent():
    """界面要能说清「筛掉了什么」，否则用户没法判断这一栏为什么是空的。"""
    posts = [
        # 够长、但不含任何领域词 → 走的应当是 domain 这条规则
        post("outside", when="2026-10-12 12:00", text="今天天气不错，我去吃了一顿火锅，很开心", likes=50),
        post("short", when="2026-10-12 11:00", text="OOTD", likes=40),
        post("old", when="2026-01-01 10:00", text="lolita 老帖记录", likes=30),
        post("ok", when="2026-10-12 10:00", text="lolita 今天的 jsk 记录，柄图很好看", likes=1),
    ]
    selection = select_hot_posts(posts, window_days=1)

    assert selection.skipped == {"window": 1, "domain": 1, "too_short": 1, "duplicate_author": 0}
    assert selection.considered == 1


def test_empty_or_undated_corpus_returns_empty_selection():
    assert select_hot_posts([]).items == []
    assert select_hot_posts([post("a", when="")]).items == []


# ------------------------------------------------------------------ 展示文本

def test_display_text_strips_tags_and_entities_and_truncates():
    raw = "<a href='https://x'>话题</a>&nbsp;正文&amp;更多" + "字" * 200
    cleaned = clean_display_text(raw, limit=20)

    assert "<a" not in cleaned and "&nbsp;" not in cleaned and "&amp;" not in cleaned
    assert "话题" in cleaned and "正文" in cleaned
    assert len(cleaned) <= 20 and cleaned.endswith("…")


def test_to_dict_exposes_everything_the_ui_needs():
    selection = select_hot_posts([post("a", when="2026-10-12 20:00", likes=3, comments=2, reposts=1)], window_days=1)
    payload = selection.to_dict()

    assert payload["anchor_label"] == "2026-10-12"
    assert payload["window_days"] == 1
    item = payload["items"][0]
    assert item["likes"] == 3 and item["comments"] == 2 and item["reposts"] == 1
    assert item["score"] == 3 + 2 * 2 + 3 * 1
    assert item["url"].startswith("https://")
    assert item["created_at"] == "2026-10-12 20:00"
