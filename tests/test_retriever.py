"""检索层测试：BM25、RRF 融合、以及向量路失败时的降级行为。"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.retriever.bm25 import BM25Retriever, tokenize
from src.retriever.hybrid import HybridRetriever
from src.schema import Chunk, Retrieved


def make_chunks() -> list[Chunk]:
    texts = {
        "c1": "JSK 是吊带裙，需要自己搭配内搭衬衫。",
        "c2": "OP 是一体式连衣裙，带袖子，穿起来省事。",
        "c3": "带亮片和烫金的柄图裙子不能机洗，建议冷水手洗。",
        "c4": "今天天气很热，我只想在家吹空调吃西瓜。",
    }
    return [
        Chunk(chunk_id=cid, post_id=cid, text=text, url=f"https://example.com/{cid}")
        for cid, text in texts.items()
    ]


def test_tokenize_drops_stopwords_and_single_letters():
    tokens = tokenize("这个 的 裙子 a 很 好看")
    assert "的" not in tokens
    assert "a" not in tokens
    assert "裙子" in tokens


def test_domain_userdict_keeps_community_terms_intact():
    """「柄图」若被切成「柄」+「图」，单字「图」会在官图/截图里到处命中，稀释 BM25 区分度。

    这条测试锁住 config/jieba_userdict.txt 的效果——词典被删或路径写错时会失败。
    """
    assert "柄图" in tokenize("柄图")
    assert tokenize("痛lo 日常化")[0] == "痛lo"
    assert "暴力撑" in tokenize("暴力撑和裙撑怎么选")
    assert "亮片" in tokenize("亮片怎么洗")


def test_latin_cjk_space_does_not_break_userdict_matching():
    """真实语料写的是「A 字撑」（带空格），空格不能把词典项 a字撑 挡掉。"""
    assert "a字撑" in tokenize("A 字撑")
    assert "a字撑" in tokenize("A字撑")
    assert "裙撑" in tokenize("裙撑先买 A 字撑")
    # 中文之间的空格不动（那是并列关系，不该被粘成一个词）
    assert tokenize("柄图 茶会") == ["柄图", "茶会"]
    assert "a字撑" in tokenize("A 字撑")


def test_userdict_file_is_shipped():
    assert Settings().jieba_userdict.exists(), "config/jieba_userdict.txt 不应缺失"


def test_bm25_finds_the_matching_document():
    retriever = BM25Retriever(make_chunks())
    hits = retriever.search("亮片裙子怎么洗", top_k=3)
    assert hits, "BM25 应该能召回与洗护相关的片段"
    assert hits[0].chunk.chunk_id == "c3"
    assert hits[0].source == "bm25"


def test_bm25_returns_nothing_for_unrelated_query():
    retriever = BM25Retriever(make_chunks())
    assert retriever.search("快速排序算法", top_k=3) == []


def test_hybrid_rrf_ranks_document_hit_by_both_routes_first():
    settings = Settings(top_k=3, recall_k=10, rrf_k=60, bm25_weight=1.0, vector_weight=1.0)
    chunks = make_chunks()

    def vector_search(query: str, top_k: int) -> list[Retrieved]:
        # 故意让向量的顺序与 BM25 不完全一致，检验融合逻辑
        order = ["c1", "c3", "c2"]
        lookup = {c.chunk_id: c for c in chunks}
        return [Retrieved(chunk=lookup[cid], score=0.9 - 0.1 * i, source="vector") for i, cid in enumerate(order[:top_k])]

    retriever = HybridRetriever(BM25Retriever(chunks), vector_search, settings)
    hits = retriever.search("亮片裙子怎么洗", top_k=3)
    assert hits[0].chunk.chunk_id == "c3"          # 两路都排前面的文档应当胜出
    assert hits[0].ranks["bm25"] == 1
    assert "bm25" in hits[0].source and "vector" in hits[0].source


def test_hybrid_survives_vector_failure():
    settings = Settings(top_k=3, min_vector_score=0.0)

    def broken_vector_search(query: str, top_k: int):
        raise RuntimeError("向量库挂了")

    retriever = HybridRetriever(BM25Retriever(make_chunks()), broken_vector_search, settings)
    hits = retriever.search("亮片裙子怎么洗", top_k=3)
    assert hits and hits[0].chunk.chunk_id == "c3"


def test_min_vector_score_filters_irrelevant_hits():
    settings = Settings(top_k=3, min_vector_score=0.5, recall_k=10)
    chunks = make_chunks()

    def vector_search(query: str, top_k: int) -> list[Retrieved]:
        return [Retrieved(chunk=chunks[3], score=0.1, source="vector")]

    retriever = HybridRetriever(BM25Retriever(chunks), vector_search, settings)
    # 低于阈值的向量结果被丢弃，而该 query 又打不中 BM25，因此应当返回空
    assert retriever.search("快速排序算法", top_k=3) == []


@pytest.mark.parametrize("top_k", [1, 2])
def test_hybrid_respects_top_k(top_k: int):
    settings = Settings(top_k=top_k, recall_k=10)
    chunks = make_chunks()

    def vector_search(query: str, k: int) -> list[Retrieved]:
        return [Retrieved(chunk=c, score=0.5, source="vector") for c in chunks[:k]]

    retriever = HybridRetriever(BM25Retriever(chunks), vector_search, settings)
    hits = retriever.search("裙子", top_k=top_k)
    assert len(hits) <= top_k
    assert all(isinstance(hit.score, float) for hit in hits)


# --------------------------------------------------------------- 闸门与融合方式
class StubBM25:
    """只返回预置结果，用来单独检验融合逻辑。

    可选地提供文档频率（``frequencies`` + ``size``）：**不提供时** HybridRetriever 会
    退回「所有内容词都算生僻词」，即不做生僻词要求——早期测试依赖这个默认行为。
    """

    def __init__(
        self,
        hits: list[Retrieved],
        frequencies: dict[str, int] | None = None,
        size: int = 0,
    ):
        self._hits = hits
        self._frequencies = frequencies or {}
        self.corpus_size = size

    def search(self, query: str, top_k: int) -> list[Retrieved]:
        return self._hits

    def document_frequency(self, token: str) -> int:
        return self._frequencies.get(token, 0)


def route_hits(pairs: list[tuple[str, float]], source: str = "bm25") -> list[Retrieved]:
    return [
        Retrieved(chunk=Chunk(chunk_id=cid, post_id=cid, text=f"{cid} 的正文"), score=score, source=source)
        for cid, score in pairs
    ]


def filler_chunks(count: int = 4) -> list[Chunk]:
    """与查询完全无关的填充文档。

    必需的：BM25Okapi 的 idf 在文档数只有 1~2 篇时会退化为 0（所有分数都是 0），
    语料太小就测不出任何 BM25 行为。
    """
    return [
        Chunk(chunk_id=f"filler{i}", post_id=f"filler{i}", text=f"第 {i} 条无关内容，讲的是天气。")
        for i in range(count)
    ]


def test_bm25_rejects_single_character_false_match():
    """实测踩过的坑：查询「怎么用 Python 写快速排序？」命中了缎带手作博文，只因里面有「用」字。"""
    chunks = [
        Chunk(chunk_id="c1", post_id="c1", text="手作发饰记录：用缎带做了一个 KC，成本不到 15 块。"),
        *filler_chunks(),
    ]
    assert BM25Retriever(chunks).search("怎么用 Python 写快速排序？", top_k=5) == []


def test_content_tokens_only_keep_multi_character_terms():
    assert BM25Retriever.content_tokens(["洗", "亮片", "python", "图"]) == {"亮片", "python"}


def test_content_token_gate_does_not_kill_single_char_queries():
    """查询只有单字时（如「洗」）内容词集合为空，闸门必须停用，否则永远查不到。"""
    chunks = [Chunk(chunk_id="c1", post_id="c1", text="洗"), *filler_chunks()]
    hits = BM25Retriever(chunks).search("洗", top_k=5)
    assert [h.chunk.chunk_id for h in hits] == ["c1"]


def test_bm25_min_ratio_prunes_weaker_matches():
    chunks = [
        Chunk(chunk_id="good", post_id="good", text="带亮片和烫金的柄图千万别机洗，建议冷水手洗。"),
        Chunk(chunk_id="weak", post_id="weak", text="今天好热啊，这条裙子一点都不想穿。"),
        *filler_chunks(),
    ]
    query = "带亮片的裙子怎么洗"

    unfiltered = BM25Retriever(chunks, min_ratio=0.0).search(query, top_k=5)
    scores = {hit.chunk.chunk_id: hit.score for hit in unfiltered}
    assert scores["good"] > scores["weak"] > 0, "「亮片」比「裙子」稀有，应当排在前面"

    threshold = min(0.99, scores["weak"] / scores["good"] + 0.05)
    filtered = BM25Retriever(chunks, min_ratio=threshold).search(query, top_k=5)
    assert [h.chunk.chunk_id for h in filtered] == ["good"]


def test_weighted_fusion_preserves_score_magnitude():
    """这是换成 weighted 的直接原因：BM25 给了 6.4 倍差距，融合后必须还在。"""
    bm25_hits = route_hits([("good", 9.17), ("weak", 1.43)])
    settings = Settings(
        fusion_mode="weighted", top_k=5, min_vector_score=0.0, bm25_weight=1.0, vector_weight=1.0
    )
    retriever = HybridRetriever(StubBM25(bm25_hits), lambda q, k: [], settings)

    hits = retriever.search("q")
    assert [h.chunk.chunk_id for h in hits] == ["good", "weak"]
    assert hits[0].score / hits[1].score > 5


def test_rrf_fusion_flattens_score_magnitude():
    """对照实验：RRF 下同样两个分数几乎同分——这就是噪声被抬进 top-k 的机制。"""
    bm25_hits = route_hits([("good", 9.17), ("weak", 1.43)])
    settings = Settings(
        fusion_mode="rrf", top_k=5, min_vector_score=0.0, rrf_k=60, bm25_weight=1.0, vector_weight=1.0
    )
    retriever = HybridRetriever(StubBM25(bm25_hits), lambda q, k: [], settings)

    hits = retriever.search("q")
    assert hits[0].score / hits[1].score < 1.05


def test_vector_floor_is_applied_before_fusion():
    def vector_search(query: str, k: int) -> list[Retrieved]:
        return [
            Retrieved(chunk=Chunk(chunk_id="high", post_id="high", text="高相似"), score=0.80, source="vector"),
            Retrieved(chunk=Chunk(chunk_id="low", post_id="low", text="低相似"), score=0.30, source="vector"),
        ]

    settings = Settings(
        fusion_mode="weighted", top_k=5, min_vector_score=0.45, bm25_weight=0.0, vector_weight=1.0
    )
    retriever = HybridRetriever(StubBM25([]), vector_search, settings)

    hits = retriever.search("q")
    assert [h.chunk.chunk_id for h in hits] == ["high"], "低于绝对阈值的结果不应进入融合"
    # 归一化以阈值为下界：恰好等于阈值 → 0 分
    assert hits[0].score == pytest.approx((0.80 - 0.45) / 0.55)


def test_document_found_by_both_routes_outranks_single_route():
    bm25_hits = route_hits([("both", 8.0), ("bm25only", 7.5)])

    def vector_search(query: str, k: int) -> list[Retrieved]:
        return [
            Retrieved(chunk=Chunk(chunk_id="both", post_id="both", text="x"), score=0.90, source="vector"),
            Retrieved(chunk=Chunk(chunk_id="veconly", post_id="veconly", text="y"), score=0.88, source="vector"),
        ]

    settings = Settings(
        fusion_mode="weighted", top_k=5, min_vector_score=0.0, bm25_weight=1.0, vector_weight=1.0
    )
    hits = HybridRetriever(StubBM25(bm25_hits), vector_search, settings).search("q")

    assert hits[0].chunk.chunk_id == "both"
    assert set(hits[0].ranks) == {"bm25", "vector"}
    assert "bm25" in hits[0].source and "vector" in hits[0].source


# ------------------------------------------- 闸门必须对每一路都生效（真实语料回归）
#
# 语料从 21 条涨到 690 条后，拒答准确率从 1.000 掉到 0.500。三个漏网案例各自暴露了
# 「闸门只作用于自己那一路」的一个侧面，下面三条测试逐一锁住。


def test_question_words_are_not_content_words():
    """「如何」这类疑问词长度 ≥2、idf 还很高，但任何文档都可能碰巧命中。"""
    assert "如何" not in tokenize("如何用 Docker 部署一个 Node 服务？")
    assert "怎么样" not in tokenize("明天北京的天气怎么样？")
    assert "哪些" not in tokenize("蜜糖甜塔包含哪些配件？")


def test_bm25_ignores_hits_that_only_match_question_words():
    """实测：该 query 的 BM25 头名是一篇日语系投稿，全靠一个「如何」命中（df=1）。"""
    chunks = [
        Chunk(chunk_id="hit", post_id="hit", text="投稿：如何把卡黄这种组合玩得更有意思？"),
        *filler_chunks(),
    ]
    assert BM25Retriever(chunks).search("如何用 Docker 部署一个 Node 服务？", top_k=5) == []


def test_bm25_hit_without_vector_support_is_dropped():
    """实测：「明天北京的天气怎么样？」被只在关键词路命中的片段带进了结果。

    该片段向量相似度只有 0.379，远低于绝对下限——关键词路不能单独把片段送进结果，
    否则 MIN_VECTOR_SCORE 形同虚设。
    """
    chunk = Chunk(chunk_id="bm25only", post_id="bm25only", text="童年的纯真 小朋友 lolita日常 北京")
    bm25_hits = [Retrieved(chunk=chunk, score=7.37, source="bm25")]

    def vector_search(query: str, k: int) -> list[Retrieved]:
        return [Retrieved(chunk=chunk, score=0.379, source="vector")]

    settings = Settings(fusion_mode="weighted", top_k=5, min_vector_score=0.44)
    hits = HybridRetriever(StubBM25(bm25_hits), vector_search, settings).search("明天北京的天气怎么样？")
    assert hits == [], "只有关键词命中、语义相似度不足的片段不应返回"


def test_vector_hit_without_any_content_word_is_dropped():
    """实测：「红烧肉怎么做才不腻？」的向量头名相似度 0.5049（高于下限），

    但正文里既没有「红烧肉」也没有「不腻」——稠密向量在短文本上的纯粹噪声。
    """
    noise = Retrieved(
        chunk=Chunk(chunk_id="noise", post_id="noise", text="调了个色，参考了老师的痛板配色。"),
        score=0.5049,
        source="vector",
    )
    settings = Settings(fusion_mode="weighted", top_k=5, min_vector_score=0.44)
    hits = HybridRetriever(StubBM25([]), lambda q, k: [noise], settings).search("红烧肉怎么做才不腻？")
    assert hits == [], "没有任何查询内容词的片段不该被当作答案依据"


def test_vector_hit_with_a_content_word_is_kept():
    """反面：同样的低相似度，只要正文真的在讲这件事，就必须留下（不能过度拒答）。"""
    good = Retrieved(
        chunk=Chunk(chunk_id="good", post_id="good", text="藤上春有两种颜色：紫、绿。"),
        score=0.4602,
        source="vector",
    )
    settings = Settings(fusion_mode="weighted", top_k=5, min_vector_score=0.44)
    hits = HybridRetriever(StubBM25([]), lambda q, k: [good], settings).search("藤上春有哪几种颜色？")
    assert [hit.chunk.chunk_id for hit in hits] == ["good"]


def test_bm25_still_works_when_the_vector_route_fails():
    """向量路整体挂掉时不能套用向量下限，否则两道闸门一起失效、检索直接不可用。"""
    chunk = Chunk(chunk_id="c3", post_id="c3", text="带亮片和烫金的柄图不能机洗，建议冷水手洗。")

    def broken(query: str, k: int):
        raise RuntimeError("向量库挂了")

    settings = Settings(fusion_mode="weighted", top_k=5, min_vector_score=0.44)
    hits = HybridRetriever(StubBM25([Retrieved(chunk=chunk, score=5.0, source="bm25")]), broken, settings).search(
        "亮片裙子怎么洗"
    )
    assert [hit.chunk.chunk_id for hit in hits] == ["c3"]


def test_two_content_word_matches_rescue_a_bm25_only_hit():
    """实测：真正相关的博文相似度也可能低于下限，此时两个独立内容词就是关键词证据。

    「谢安然今天穿的是哪个牌子？」对应的博文同时命中「安然」和「今天」，
    若严格按向量下限一刀切会被误拒（这是补上闸门后实际发生的回归）。
    """
    chunk = Chunk(
        chunk_id="anran",
        post_id="anran",
        text="转 关 啾条【少女人形 天文钟】谢安然超话 安然今天穿的是 古典玩偶 lolita",
    )
    bm25_hits = [Retrieved(chunk=chunk, score=21.6, source="bm25")]

    def vector_search(query: str, k: int) -> list[Retrieved]:
        return [Retrieved(chunk=chunk, score=0.42, source="vector")]   # 低于 0.44

    settings = Settings(fusion_mode="weighted", top_k=5, min_vector_score=0.44)
    hits = HybridRetriever(StubBM25(bm25_hits), vector_search, settings).search("谢安然今天穿的是哪个牌子？")
    assert [hit.chunk.chunk_id for hit in hits] == ["anran"]


def test_single_content_word_match_does_not_rescue():
    """对照：只匹配一个泛词（「北京」）不算证据，仍要被向量下限拦掉。"""
    chunk = Chunk(chunk_id="weather", post_id="weather", text="童年的纯真 小朋友 lolita日常 北京")
    bm25_hits = [Retrieved(chunk=chunk, score=7.37, source="bm25")]

    def vector_search(query: str, k: int) -> list[Retrieved]:
        return [Retrieved(chunk=chunk, score=0.379, source="vector")]

    settings = Settings(fusion_mode="weighted", top_k=5, min_vector_score=0.44)
    hits = HybridRetriever(StubBM25(bm25_hits), vector_search, settings).search("明天北京的天气怎么样？")
    assert hits == []


# ------------------------------------------- 单术语查询 / 生僻词 / 短文本（第二轮真实回归）
#
# 第二轮体检用 12 条真实提问打出三个问题：
#   1) 「KC 是什么」「柄图是什么」被**系统性全部拒答**（内容词只有 1 个，永远达不到「≥2」）；
#   2) 「藤上春这条裙子多少钱」的 top-5 里 4 条只命中了「裙子」；指示代词「这条」df=9
#      还被误判成生僻词，把无关博文留了下来；
#   3) 「亮片和烫金的裙子怎么洗？」里的「亮片」在语料里 df=0，被当成「必须命中」，
#      结果连真正讲洗护的博文都被误杀。


def test_single_content_word_query_is_not_refused():
    """内容词只有一个时，「命中了那个词」就是最强证据——不能因为凑不够 2 个而拒答。"""
    chunk = Chunk(chunk_id="kc", post_id="kc", text="白雪童话 KC&头冠KC 施华洛版 水蓝&粉色&黄色&天蓝 人台展示")
    bm25_hits = [Retrieved(chunk=chunk, score=9.0, source="bm25")]

    def vector_search(query: str, k: int) -> list[Retrieved]:
        return [Retrieved(chunk=chunk, score=0.30, source="vector")]   # 低于下限

    settings = Settings(fusion_mode="weighted", top_k=5, min_vector_score=0.44)
    hits = HybridRetriever(StubBM25(bm25_hits), vector_search, settings).search("KC 是什么")
    assert [hit.chunk.chunk_id for hit in hits] == ["kc"]


def test_demonstratives_are_not_content_words():
    """「这条」「那种」没有区分度，与出现频率无关；df 低也不能当生僻词用。"""
    assert "这条" not in tokenize("藤上春这条裙子多少钱")
    assert "那种" not in tokenize("那种裙子怎么洗")
    assert "藤上" in tokenize("藤上春这条裙子多少钱")


def test_hit_matching_only_a_common_word_is_dropped():
    """实测：藤上春的 top-5 里有 4 条只命中了「裙子」（df=98/615）。"""
    target = Chunk(chunk_id="target", post_id="target", text="裙子：藤上春 Time：10.20 Price：jsk=295")
    other = Chunk(chunk_id="other", post_id="other", text="浪漫蔷薇 紫色裙子 心情变好就来试裙子了 真的好日常")
    bm25_hits = [Retrieved(chunk=target, score=9.0, source="bm25"), Retrieved(chunk=other, score=8.0, source="bm25")]
    frequencies = {"藤上": 1, "裙子": 98, "钱": 40}

    def vector_search(query: str, k: int) -> list[Retrieved]:
        # 两条都没有语义支持，逼着生僻词闸门来区分
        return [
            Retrieved(chunk=target, score=0.30, source="vector"),
            Retrieved(chunk=other, score=0.28, source="vector"),
        ]

    settings = Settings(fusion_mode="weighted", top_k=5, min_vector_score=0.44, distinctive_df_ratio=0.10)
    retriever = HybridRetriever(
        StubBM25(bm25_hits, frequencies=frequencies, size=615), vector_search, settings
    )
    hits = retriever.search("藤上春这条裙子多少钱")
    assert [hit.chunk.chunk_id for hit in hits] == ["target"]


def test_unmatchable_query_term_does_not_block_results():
    """「亮片」在语料里 df=0，任何片段都不可能命中——不能拿它当必须条件。"""
    washing = Chunk(chunk_id="wash", post_id="wash", text="洗的时候一定要套上洗衣袋清洗喔，更好的保护裙子")
    bm25_hits = [Retrieved(chunk=washing, score=6.0, source="bm25")]
    frequencies = {"亮片": 0, "烫金": 1, "裙子": 98}

    def vector_search(query: str, k: int) -> list[Retrieved]:
        return [Retrieved(chunk=washing, score=0.52, source="vector")]   # 有语义支持

    settings = Settings(fusion_mode="weighted", top_k=5, min_vector_score=0.44, distinctive_df_ratio=0.10)
    retriever = HybridRetriever(
        StubBM25(bm25_hits, frequencies=frequencies, size=615), vector_search, settings
    )
    hits = retriever.search("亮片和烫金的裙子怎么洗？")
    assert [hit.chunk.chunk_id for hit in hits] == ["wash"]


def test_semantic_support_substitutes_for_a_distinctive_word():
    """语文上讲的是同一件事、只是用词不同的片段，不该被生僻词闸门误杀。"""
    reworded = Chunk(chunk_id="reworded", post_id="reworded", text="裙子上的装饰别机洗，套洗衣袋冷水手洗")
    bm25_hits = [Retrieved(chunk=reworded, score=5.0, source="bm25")]
    frequencies = {"烫金": 1, "裙子": 98}

    def vector_search(query: str, k: int) -> list[Retrieved]:
        return [Retrieved(chunk=reworded, score=0.50, source="vector")]

    settings = Settings(fusion_mode="weighted", top_k=5, min_vector_score=0.44, distinctive_df_ratio=0.10)
    retriever = HybridRetriever(
        StubBM25(bm25_hits, frequencies=frequencies, size=615), vector_search, settings
    )
    assert retriever.search("烫金的裙子怎么洗？") != []


def test_query_about_a_concept_absent_from_the_corpus_is_refused():
    """在域内但无答案的提问：向量下限在这里彻底失效，只能靠「核心概念不存在」识别。

    实测「怎么分辨 lolita 的山正？」返回 5 篇随便什么 lolita 博文，原始相似度 0.52——
    因为整个语料都是 lolita 内容。而「山正」「分辨」在 615 个片段里 df=0，剩下能匹配的
    只有 df=367 的「lolita」。此时返回任何东西都是硬答。
    """
    generic = Chunk(chunk_id="generic", post_id="generic", text="LOLITA 镜中花这个收腰真的太好了吧，花墙柄图就是最绝的")
    bm25_hits = [Retrieved(chunk=generic, score=6.0, source="bm25")]
    frequencies = {"lolita": 367}

    def vector_search(query: str, k: int) -> list[Retrieved]:
        # 相似度还挺高——但这是「整个语料都是 lolita」的假象，不是相关性
        return [Retrieved(chunk=generic, score=0.5215, source="vector")]

    settings = Settings(fusion_mode="weighted", top_k=5, min_vector_score=0.44, distinctive_df_ratio=0.10)
    retriever = HybridRetriever(
        StubBM25(bm25_hits, frequencies=frequencies, size=615), vector_search, settings
    )
    assert retriever.search("怎么分辨 lolita 的山正？") == []


def test_absent_term_with_a_distinctive_anchor_is_not_refused():
    """对照：缺「亮片」但还有「烫金」可锚定，属于**可以**回答的提问，不能拒答。"""
    washing = Chunk(chunk_id="wash", post_id="wash", text="裙子洗的时候要套洗衣袋，刺绣贴下水会变深")
    bm25_hits = [Retrieved(chunk=washing, score=6.0, source="bm25")]
    frequencies = {"亮片": 0, "烫金": 1, "裙子": 98}

    def vector_search(query: str, k: int) -> list[Retrieved]:
        return [Retrieved(chunk=washing, score=0.52, source="vector")]

    settings = Settings(fusion_mode="weighted", top_k=5, min_vector_score=0.44, distinctive_df_ratio=0.10)
    retriever = HybridRetriever(
        StubBM25(bm25_hits, frequencies=frequencies, size=615), vector_search, settings
    )
    assert [hit.chunk.chunk_id for hit in retriever.search("亮片和烫金的裙子怎么洗？")] == ["wash"]
