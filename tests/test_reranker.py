"""精排（cross-encoder）的测试。

精排的定位是**加分项**，所以测试的重心不是「它排得准不准」（那要靠实测指标），
而是两条工程底线：

1. **任何失败都不能让检索挂掉**：模型加载失败、打分抛异常、候选太少——
   都必须退回原顺序，而不是抛错或返回空；
2. **它只重排已经通过闸门的候选**：拒答行为不能被它改变（空必须还是空）。

第 2 条尤其重要：精排如果插在闸门**之前**，就会把被闸门挡掉的噪声重新捞回结果里。
"""

from __future__ import annotations

from types import SimpleNamespace

from src.config import Settings
from src.retriever.hybrid import HybridRetriever
from src.retriever.reranker import Reranker
from src.schema import Chunk, Retrieved


def make_hits(count: int = 6) -> list[Retrieved]:
    hits: list[Retrieved] = []
    for index in range(1, count + 1):
        hits.append(
            Retrieved(
                chunk=Chunk(
                    chunk_id=f"p{index}#0",
                    post_id=f"p{index}",
                    text=f"第 {index} 条内容：关于柄图与裙撑的说明。",
                ),
                score=1.0 / index,          # 原顺序：p1 > p2 > ...
                source="bm25+vector",
                ranks={"bm25": index, "vector": index},
            )
        )
    return hits


class FakeModel:
    """假 cross-encoder：按片段编号给分，可指定成「完全倒序」。"""

    def __init__(self, scores=None, exc: Exception | None = None):
        self.scores = scores
        self.exc = exc
        self.calls: list[list] = []

    def predict(self, pairs, batch_size=None):
        self.calls.append(list(pairs))
        if self.exc:
            raise self.exc
        if self.scores is not None:
            return self.scores
        # 默认：编号越大分越高（正好把原顺序倒过来）
        return [float(index) for index in range(len(pairs))]


def make_settings(**overrides) -> Settings:
    overrides.setdefault("reranker_enabled", True)
    return Settings(**overrides)


# ------------------------------------------------------------------ 基本行为

def test_reranker_reorders_by_cross_encoder_score():
    # 给 6 条候选打「越靠后分越高」，精排后应当整体倒序
    model = FakeModel(scores=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0][::-1])
    reranker = Reranker(make_settings(), model=model)

    result = reranker.rerank("柄图", make_hits(6), top_k=3)

    assert [hit.chunk.chunk_id for hit in result] == ["p6#0", "p5#0", "p4#0"]
    assert len(model.calls) == 1
    assert model.calls[0][0][0] == "柄图", "要把查询原文交给 cross-encoder"


def test_score_is_normalised_to_zero_one():
    """cross-encoder 给的是 logit，要压成 0~1，否则界面上的「相关度」没法看。"""
    reranker = Reranker(make_settings(), model=FakeModel(scores=[0.0, 100.0, -100.0]))
    result = reranker.rerank("q", make_hits(3), top_k=3)
    for hit in result:
        assert 0.0 <= hit.score <= 1.0
    assert result[0].score > 0.99


def test_ranks_and_source_record_the_rerank_position():
    reranker = Reranker(make_settings(), model=FakeModel(scores=[0.0, 1.0, 2.0]))
    result = reranker.rerank("q", make_hits(3), top_k=3)
    assert [hit.ranks["rerank"] for hit in result] == [1, 2, 3]
    assert all("rerank" in hit.source for hit in result)


def test_disabled_reranker_returns_original_order_without_calling_model():
    model = FakeModel()
    reranker = Reranker(make_settings(reranker_enabled=False), model=model)
    hits = make_hits(6)

    result = reranker.rerank("q", hits, top_k=3)

    assert [hit.chunk.chunk_id for hit in result] == ["p1#0", "p2#0", "p3#0"]
    assert model.calls == [], "关闭时不该加载/调用模型"


def test_candidates_fewer_than_two_skip_the_model():
    model = FakeModel()
    reranker = Reranker(make_settings(), model=model)

    result = reranker.rerank("q", make_hits(1), top_k=5)

    assert len(result) == 1
    assert model.calls == [], "只有一条候选，没有可重排的空间"


def test_reranking_still_happens_when_candidate_count_equals_top_k():
    """候选恰好等于 top_k 时**也要**精排：顺序仍然影响 MRR。"""
    model = FakeModel(scores=[0.0, 1.0, 2.0])
    reranker = Reranker(make_settings(), model=model)

    result = reranker.rerank("q", make_hits(3), top_k=3)

    assert [hit.chunk.chunk_id for hit in result] == ["p3#0", "p2#0", "p1#0"]
    assert model.calls, "不该因为「数量够了」就跳过精排"


# ------------------------------------------------------------------ 降级

def test_predict_failure_keeps_original_order():
    reranker = Reranker(make_settings(), model=FakeModel(exc=RuntimeError("显存不足")))
    result = reranker.rerank("q", make_hits(6), top_k=3)
    assert [hit.chunk.chunk_id for hit in result] == ["p1#0", "p2#0", "p3#0"]


def test_model_load_failure_degrades_instead_of_raising(monkeypatch):
    """模型文件缺失/离线/内存不足都不能让检索失败——精排只是加分项。"""

    class BrokenCrossEncoder:
        def __init__(self, *args, **kwargs):
            raise OSError("模型文件不存在")

    monkeypatch.setattr("sentence_transformers.CrossEncoder", BrokenCrossEncoder, raising=False)
    reranker = Reranker(make_settings())          # 不注入 model → 走真实加载路径
    result = reranker.rerank("q", make_hits(6), top_k=2)

    assert [hit.chunk.chunk_id for hit in result] == ["p1#0", "p2#0"]
    assert reranker._load_failed is True


def test_reranker_top_n_brings_in_deeper_candidates():
    """`RERANKER_TOP_N` 的价值：把融合里排得靠后、但其实是答案的片段捞上来。

    这里 8 条候选里给最后一条（融合第 8 名）最高分，它必须能进最终的 top-2。
    """
    scores = [0.0] * 7 + [99.0]
    reranker = Reranker(make_settings(reranker_top_n=8), model=FakeModel(scores=scores))

    result = reranker.rerank("q", make_hits(8), top_k=2)

    assert result[0].chunk.chunk_id == "p8#0", "深处的候选要被精排捞上来"
    assert len(result) == 2


def test_head_always_covers_top_k_even_if_top_n_is_smaller():
    """`RERANKER_TOP_N` 是**下限而非上限**：它小于 top_k 时不能少返回结果。"""
    reranker = Reranker(make_settings(reranker_top_n=1), model=FakeModel(scores=[2.0, 1.0, 0.0]))

    result = reranker.rerank("q", make_hits(6), top_k=3)

    assert len(result) == 3


def test_score_count_mismatch_keeps_every_candidate():
    """模型返回的分数个数对不上时，宁可原序返回也**不能丢候选**。

    `zip` 会静默截断 —— 那是最难发现的一类 bug（结果看起来正常，只是少了几条）。
    """
    reranker = Reranker(make_settings(), model=FakeModel(scores=[1.0, 0.0]))  # 只给 2 个分，却有 6 条
    result = reranker.rerank("q", make_hits(6), top_k=4)

    assert len(result) == 4
    assert [hit.chunk.chunk_id for hit in result] == ["p1#0", "p2#0", "p3#0", "p4#0"]


# ------------------------------------------------------------------ 与闸门的关系

class FakeBM25:
    """关键词路：返回两条命中查询内容词的候选。"""

    def __init__(self):
        self.min_ratio = 0.2

    def search(self, query, top_k):
        return [
            Retrieved(
                chunk=Chunk(chunk_id="keep1#0", post_id="keep1", text="柄图 与 裙撑 的 说明"),
                score=5.0,
            ),
            Retrieved(
                chunk=Chunk(chunk_id="keep2#0", post_id="keep2", text="柄图 搭配 裙撑 的 经验"),
                score=4.0,
            ),
        ]


def test_reranker_only_sees_candidates_that_passed_the_gates():
    """精排必须插在闸门**之后**：否则被闸门挡掉的噪声会被重新排回结果里。"""
    seen: list[str] = []

    class SpyModel(FakeModel):
        def predict(self, pairs, batch_size=None):
            seen.extend(text for _, text in pairs)
            return [1.0] * len(pairs)

    def vector_search(query, top_k):
        # 这条虽然相似度高，但没有命中任何查询内容词 → 应当被内容词闸门挡掉
        return [
            Retrieved(
                chunk=Chunk(chunk_id="noise#0", post_id="noise", text="红烧肉怎么做才不腻"),
                score=0.99,
            )
        ]

    retriever = HybridRetriever(
        FakeBM25(), vector_search, make_settings(), Reranker(make_settings(), model=SpyModel())
    )
    hits = retriever.search("柄图 裙撑", top_k=5)

    assert {hit.chunk.chunk_id for hit in hits} == {"keep1#0", "keep2#0"}, "噪声不该被精排捞回来"
    assert seen and all("红烧肉" not in text for text in seen), "精排只该看到通过闸门的候选"


def test_reranker_cannot_turn_refusal_into_results():
    """空结果就是空结果：精排不该把「没有资料」变成「有资料」。"""
    bm25 = FakeBM25()

    def empty_vector(query, top_k):
        return []

    class AlwaysEmpty:
        min_ratio = 0.2

        def search(self, query, top_k):
            return []

    retriever = HybridRetriever(
        AlwaysEmpty(), empty_vector, make_settings(), Reranker(make_settings(), model=FakeModel())
    )
    assert retriever.search("完全不存在的话题", top_k=5) == []
