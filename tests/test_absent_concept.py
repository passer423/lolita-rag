"""「问到语料之外」判定的测试。

补全长博文之后语料从 1372 段涨到 1642 段，暴露了一个真实回退：
「考研数学该怎么复习？」只因为「考研」在语料里出现过 **1 次**（还在一篇无关的粉丝长文里）
就被当成锚点，于是返回了一篇个人随笔；「飞机票怎么改签？」同理（改签 df=2）。

所以新增第二条判据：**多数内容词都在语料里缺失 → 直接拒答**。
阈值 0.5 是量出来的：32 道评测题里 24 道正例的最大缺失比只有 0.33。
"""

from __future__ import annotations

from src.config import Settings
from src.retriever.hybrid import HybridRetriever
from src.schema import Chunk, Retrieved


class FakeBM25WithStats:
    """带语料统计的假 BM25：只提供 document_frequency / corpus_size。"""

    corpus_size = 1000

    def __init__(self, dfs: dict[str, int], hits=None):
        self.min_ratio = 0.2
        self.dfs = dfs
        self.hits = hits or []

    def document_frequency(self, token: str) -> int:
        return self.dfs.get(token, 0)

    def search(self, query: str, top_k: int):
        return list(self.hits)


def build(dfs: dict[str, int], *, hits=None, vector=None, **settings_kw):
    settings = Settings(**settings_kw)
    retriever = HybridRetriever(
        FakeBM25WithStats(dfs, hits), lambda query, k: list(vector or []), settings
    )
    return retriever, settings


def hit(chunk_id: str, text: str, score: float = 0.9) -> Retrieved:
    return Retrieved(
        chunk=Chunk(chunk_id=chunk_id, post_id=chunk_id, text=text), score=score, source="vector"
    )


# ------------------------------------------------------------------ 单元

def test_single_absent_concept_without_anchor_refuses():
    """原规则：缺失的概念 + 没有有区分度的词 → 拒答（「消毒」「红烧肉」都是这一类）。"""
    retriever, _ = build({"裙子": 245, "lolita": 873})
    assert retriever._too_much_absent({"lolita", "裙子", "消毒"}, {"消毒"}, set())


def test_majority_absent_refuses_even_if_one_term_anchors():
    """新规则：3 个内容词里 2 个不存在 → 拒答，哪怕第 3 个能当锚点。

    这正是「考研数学该怎么复习」的情形（考研:1 数学:0 复习:0）。
    """
    retriever, _ = build({"考研": 1, "裙子": 245})
    assert retriever._too_much_absent({"考研", "数学", "复习"}, {"数学", "复习"}, {"考研"})


def test_half_absent_refuses_at_the_boundary():
    """「飞机票怎么改签」= 2 个内容词里 1 个缺失，正好在边界上，应当拒答。"""
    retriever, _ = build({"改签": 2})
    assert retriever._too_much_absent({"飞机票", "改签"}, {"飞机票"}, {"改签"})


def test_in_domain_queries_are_not_refused():
    """正例的最大缺失比是 0.33（「藤上春有哪几种颜色？」里的分词噪声），不能被误伤。"""
    retriever, _ = build({"颜色": 44})
    assert not retriever._too_much_absent({"藤上", "春有", "颜色"}, {"春有"}, {"藤上"})


def test_absent_majority_ratio_is_tunable():
    _, settings = build({"考研": 1})
    assert settings.absent_majority_ratio == 0.5

    retriever, _ = build({"考研": 1}, absent_majority_ratio=1.0)
    assert not retriever._too_much_absent({"考研", "数学", "复习"}, {"数学", "复习"}, {"考研"})


def test_no_content_words_never_refuses():
    retriever, _ = build({})
    assert not retriever._too_much_absent(set(), set(), set())


# ------------------------------------------------------------------ 端到端（假检索器）

def test_out_of_domain_query_returns_empty_even_though_a_hit_exists():
    """就算向量路真的返回了一条（匹配了那个「只出现一次」的词），也要拒答。"""
    dfs = {"考研": 1}
    candidate = hit("p1#0", "我的理想依然只是嘴上说说而已 考研 这条路")
    retriever, _ = build(dfs, vector=[candidate])

    assert retriever.search("考研数学该怎么复习", top_k=5) == []


def test_same_hit_is_returned_when_the_rule_is_turned_off():
    """把阈值调到 1.0 关掉这条规则，同一条命中就会返回——证明拦截确实来自这条规则。"""
    dfs = {"考研": 1}
    candidate = hit("p1#0", "我的理想依然只是嘴上说说而已 考研 这条路")
    retriever, _ = build(dfs, vector=[candidate], absent_majority_ratio=1.0)

    hits = retriever.search("考研数学该怎么复习", top_k=5)

    assert [item.chunk.chunk_id for item in hits] == ["p1#0"]
