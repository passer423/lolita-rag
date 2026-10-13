"""「语义支持」窗口的回归测试。

## 这个坑为什么值得单独一个文件

语料从 1642 涨到 1672 段之后，「藤上春有哪几种颜色？」从**第 1 名变成返回空**。
根因不是新数据不相关，而是：

* 那条标注帖是 BM25 第 1 名，但它**掉出了向量路的前 20 名**；
* 「有没有语义支持」只在向量路的召回窗口内可查 —— 查不到就被当成 0；
* 于是闸门判定它「既没有语义支持、关键词证据也不够」，整题被丢掉。

**它自己的相似度一点没变，变的是「还在不在窗口里」。** 而语料现在每晚都在增长
（`scripts/daily_update.py`），所以这种窗口边缘的误杀会反复出现，必须锁住。

下面既有正向用例（宽窗口下应当命中），也有**反向对照**（把窗口调回 20 就复现 bug）
—— 没有反向对照，这个测试可能只是在验证「实现恰好能跑」。
"""

from __future__ import annotations

from src.config import Settings
from src.retriever.hybrid import HybridRetriever
from src.schema import Chunk, Retrieved

QUERY = "藤上春有哪几种颜色"


class FakeBM25:
    """关键词路：固定返回命中的那条；带语料统计供闸门使用。"""

    corpus_size = 1672

    def __init__(self, hits: list[Retrieved], frequencies: dict[str, int]):
        self.min_ratio = 0.2
        self._hits = hits
        self._frequencies = frequencies

    def search(self, query: str, top_k: int) -> list[Retrieved]:
        return list(self._hits)

    def document_frequency(self, token: str) -> int:
        return self._frequencies.get(token, 0)


def labeled_chunk() -> Chunk:
    return Chunk(
        chunk_id="label#0",
        post_id="label",
        text="lolita上新 裙子：藤上春 Time：10.20 Price：jsk=295 Color：紫、绿",
    )


def build(*, support_k: int, labeled_vector_rank: int = 40, labeled_score: float = 0.60):
    """标注帖在语料里的位置：BM25 第 1 名，向量第 ``labeled_vector_rank`` 名。"""
    chunk = labeled_chunk()
    bm25 = FakeBM25(
        [Retrieved(chunk=chunk, score=9.0, source="bm25")],
        {"藤上": 1, "春有": 0, "颜色": 46},
    )
    # 前面的干扰项与查询毫无关系（内容词一个都不命中）
    fillers = [
        Retrieved(
            chunk=Chunk(chunk_id=f"filler{i}", post_id=f"filler{i}", text="第一条无关内容，讲的是别的"),
            score=0.62 - i * 0.001,
            source="vector",
        )
        for i in range(labeled_vector_rank - 1)
    ]
    ordered = fillers + [Retrieved(chunk=chunk, score=labeled_score, source="vector")]
    settings = Settings(
        recall_k=20,
        semantic_support_k=support_k,
        min_vector_score=0.44,
        top_k=5,
    )
    return HybridRetriever(bm25, lambda query, k: ordered[:k], settings), chunk


def test_labeled_hit_survives_when_it_is_outside_the_recall_window():
    """正向：标注帖在向量第 40 名（recall_k=20 之外），但相似度 0.60 够高 → 必须命中。"""
    retriever, chunk = build(support_k=200)

    hits = retriever.search(QUERY, top_k=5)

    assert [item.chunk.chunk_id for item in hits] == [chunk.chunk_id]


def test_narrow_support_window_reproduces_the_bug():
    """反向对照：把语义支持窗口压回 recall_k=20，同一道题就会返回空。

    这条用例是给修复本身做的对照——证明上面那条不是「实现恰好能跑」。
    """
    retriever, _ = build(support_k=20)

    assert retriever.search(QUERY, top_k=5) == [], "窗口太窄时确实会把正确结果误杀"


def test_default_support_window_is_wider_than_recall():
    """默认值必须比 recall_k 宽，否则这个 bug 会在没人注意时回来。"""
    settings = Settings()
    assert settings.semantic_support_k > settings.recall_k


def test_support_window_does_not_widen_the_fusion_candidates():
    """宽窗口**只用于查询语义支持**，参与融合的仍只有前 recall_k 条。

    否则就变成悄悄改了召回行为——那是另一件事，得单独测、单独承担指标变化。
    """
    chunk = labeled_chunk()
    bm25 = FakeBM25([], {"藤上": 1, "春有": 0, "颜色": 46})
    # 前 20 名相似度低于下限（不该进结果），第 30 名是一个高分项
    ordered = [
        Retrieved(
            chunk=Chunk(chunk_id=f"filler{i}", post_id=f"filler{i}", text="第一条无关内容，讲的是别的"),
            score=0.20,
            source="vector",
        )
        for i in range(29)
    ] + [Retrieved(chunk=chunk, score=0.90, source="vector")]
    settings = Settings(recall_k=20, semantic_support_k=200, min_vector_score=0.44, top_k=5)
    retriever = HybridRetriever(bm25, lambda query, k: ordered[:k], settings)

    # 第 30 名虽然分数高，但它不在融合窗口里 → 仍然不该被召回
    assert retriever.search(QUERY, top_k=5) == []
