"""从语料挖真实提问的测试。

用**真实语料里的原文**当用例（下面这些字符串都是从 `chunks.jsonl` 里抄出来的），
而不是我编的漂亮句子——挖查询这件事最容易在真实口语上翻车。
"""

from __future__ import annotations

from src.pipeline.real_queries import (
    extract_query,
    looks_like_question,
    mine_queries,
    summarize_queries,
)
from src.schema import Chunk


def chunk(text: str, cid: str = "c1") -> Chunk:
    return Chunk(chunk_id=cid, post_id=cid, text=text)


# ------------------------------------------------------------------ 是不是提问

def test_real_help_posts_are_questions():
    assert looks_like_question("【求助】仓鼠lo的大家是怎么调理的呢。买了很多日常款的裙子舍不得穿")
    assert looks_like_question("有没有老婆推荐一下lo裙 新手第一次买")
    assert looks_like_question("怎么分辨山正啊 看不出来 印刷倒是挺清晰的")


def test_product_and_showoff_posts_are_not_questions():
    assert not looks_like_question("lolita上新 lolita洋装上新 电阻： 裙子：藤上春 Time：10.20 Price：jsk=295")
    assert not looks_like_question("花与珍珠匣三丽鸥家族合作系列人台展示 粉嫩嫩的桃花漫天飘呀飘")
    assert not looks_like_question("这也太美了吧？")          # 有问号但是感叹
    assert not looks_like_question("")
    # 「什么…都/也」是陈述句：实测从语料里挖出过「lolita出物 什么裙子都收」这种假提问
    assert not looks_like_question("lolita出物 什么裙子都收 走咸鱼 s码的m码")
    assert not looks_like_question("怎么搭配都好看")


# ------------------------------------------------------------------ 抽查询

def test_extracts_the_polite_request_as_a_query():
    query = extract_query("【投稿】 递遮阳伞，想问问大家葡萄园翻领长op和迷宫花园长op在夏天的对比？哪件更凉快呢？")
    assert query is not None
    assert "葡萄园" in query
    assert "【投稿】" not in query, "结构性包装要剥掉（壳之间还有空格）"
    assert len(query) <= 42
    # 口语寒暄（「递遮阳伞，」）**故意**保留：它属于真实分布，
    # 只剥结构包装；把某条样本的措辞硬编码进正则就是过拟合。


def test_strips_topic_and_mention_wrappers():
    assert extract_query("#Lolita[超话]#怎么分辨山正啊？") == "怎么分辨山正啊？"
    assert extract_query("@嘀咕娘 请问这条裙子怎么洗？") == "请问这条裙子怎么洗？"


def test_rejects_queries_that_are_too_short_or_too_long():
    assert extract_query("有没有？") is None                     # 太短，没有信息
    assert extract_query("请问" + "很长的描述" * 20 + "？") is None  # 太长，不是一句查询


def test_returns_none_for_non_questions():
    assert extract_query("lolita出物 自带价格 送斗篷内搭裙撑m") is None


# ------------------------------------------------------------------ 批量挖

def test_mine_queries_dedupes_and_respects_limit():
    chunks = [
        chunk("有没有老婆推荐一下lo裙 新手第一次买", "c1"),
        chunk("有没有老婆推荐一下lo裙 新手第一次买", "c2"),      # 重复
        chunk("怎么分辨山正啊 看不出来", "c3"),
        chunk("lolita上新 裙子：藤上春", "c4"),                   # 不是提问
        chunk("请问茶会礼仪要注意什么？", "c5"),
    ]
    queries = mine_queries(chunks, limit=10)
    assert queries == [
        "有没有老婆推荐一下lo裙 新手第一次买",
        "怎么分辨山正啊 看不出来",
        "请问茶会礼仪要注意什么？",
    ]
    assert mine_queries(chunks, limit=2) == queries[:2]


def test_summarize_handles_empty_and_non_empty():
    assert "没有挖到" in summarize_queries([])
    text = summarize_queries(["a" * 8, "b" * 8], limit=1)
    assert "共 2 条" in text
    assert text.count("·") == 1
