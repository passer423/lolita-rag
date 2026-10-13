"""混合检索：BM25 与向量召回融合。

## 为什么默认不是 RRF

最初这里用的是 RRF（Reciprocal Rank Fusion），理由是它对两路的分数尺度不敏感、
几乎免调参。但实测发现它在**小候选集**下会坏事：

    查询「带亮片的裙子应该怎么洗？」
      BM25  : 正确博文 9.17 分，另一篇只匹配了「裙子」的拿 1.43 分 —— 差 6.4 倍
      RRF   : 1/61 = 0.01639 与 1/62 = 0.01613 —— 差 1.6%，几乎同分

于是「只匹配了一个常见词」的文档被抬进 top-k，用户看到的就是一堆不相关内容。
RRF 的压平特性在大候选池里是优点（名次本身可靠），在这里却是信息损失。

所以默认改成 **weighted**：每路先按本查询内的最高分归一化，再线性加权，
保留「好多少」的信息。RRF 仍然保留（``FUSION_MODE=rrf``）以备对比。
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.config import Settings, settings as default_settings
from src.retriever.bm25 import BM25Retriever, tokenize
from src.schema import Chunk, Retrieved

logger = logging.getLogger(__name__)

# 关键词路要**独立**把片段送进结果（没有向量支持时），必须满足下面之一：
#   * 命中 ≥ 该数量的不同查询内容词；
#   * 或者把查询的**全部**内容词都命中（``matched == len(content)``）。
#
# 为什么是 2：「单个词太容易是巧合」。实测「明天北京的天气怎么样？」的命中片段各只匹配
# 一个泛词（「北京」df=3、「明天」df=4），相似度 0.36~0.38，全是噪声；而真正相关的
# 「谢安然今天穿的是哪个牌子？」对应的博文同时命中「安然」与「今天」，相似度却低于
# 向量下限——只按向量下限一刀切会把这条误拒。
#
# 「全覆盖」这一条是被拒答 bug 逼出来的：**内容词只有一个的查询永远达不到数量门槛**
# （1 < 2），于是「KC 是什么」「柄图是什么」这类查询被系统性地全部拒答，而语料里
# 明明有 5 条在讲这些词。单词查询下「命中了唯一那个词」本身就是最强的证据。
BM25_ONLY_MIN_CONTENT_MATCHES = 2


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        vector_search,
        settings: Settings | None = None,
        reranker=None,
    ):
        """``vector_search`` 是一个 ``(query, top_k) -> list[Retrieved]`` 的可调用对象。

        这样 HybridRetriever 不需要知道 embedding 与向量库的任何细节，便于测试。

        ``reranker`` 可注入（``None`` 表示不精排）；只在**闸门之后**参与，见 `search`。
        """
        self.bm25 = bm25
        self.vector_search = vector_search
        self.settings = settings or default_settings
        self.reranker = reranker
        # 片段分词结果按需计算并缓存。不直接复用 BM25 索引里的分词，
        # 是为了让本类仍然只依赖 bm25 的 ``search(query, top_k)`` 一个接口
        # （测试里可以注入任意假实现）。
        self._chunk_tokens: dict[str, set[str]] = {}

    def _chunk_content_tokens(self, chunk: Chunk) -> set[str]:
        cached = self._chunk_tokens.get(chunk.chunk_id)
        if cached is None:
            cached = set(tokenize(chunk.text))
            self._chunk_tokens[chunk.chunk_id] = cached
        return cached

    def _query_term_profile(self, content: set[str]) -> tuple[set[str], set[str]]:
        """把查询内容词分成两组：``(有区分度的词, 语料里根本没有的词)``。

        三个刻意的处理：

        * **df = 0 的词不能进「必须命中」集合**。它们在语料里根本不存在，任何片段都
          不可能命中，拿它们当条件会让查询直接查空。实测「亮片和烫金的裙子怎么洗？」
          里的「亮片」df=0，于是连真正讲洗护的博文都被误杀。
        * 但**df = 0 本身是有用的信号**，所以要单独返回：它说明用户问的概念语料里没有。
        * 拿不到语料统计时（例如测试注入的假 BM25）退回「全都是生僻词」，
          等价于旧行为，不会因为缺少统计信息而过度过滤。
        """
        if not content:
            return set(), set()
        frequency = getattr(self.bm25, "document_frequency", None)
        size = getattr(self.bm25, "corpus_size", 0)
        if not callable(frequency) or not size:
            return set(content), set()
        limit = max(1, int(size * self.settings.distinctive_df_ratio))
        distinctive: set[str] = set()
        absent: set[str] = set()
        for token in content:
            count = frequency(token)
            if count == 0:
                absent.add(token)
            elif count <= limit:
                distinctive.add(token)
            # 计数为 0 或 <= limit 之外的词（即太常见）都不进 distinctive
        return distinctive, absent

    def _too_much_absent(self, content: set[str], absent: set[str], distinctive: set[str]) -> bool:
        """判断这条查询是不是**问到了语料之外**，应当直接拒答。

        两条规则，都是被真实数据打出来的：

        1. 有词在语料里根本不存在（df=0），且**没有任何有区分度的词能锚定**它
           —— 原来的规则。
        2. **多数内容词都不存在**（默认 ≥ 1/2）—— 补全长博文后语料从 1372 涨到 1642 段，
           实测暴露的新问题：「考研数学该怎么复习？」里只有「考研」在语料里出现过 **1 次**
           （还在一篇无关的粉丝长文里），却被当成锚点，于是返回了一篇个人随笔；
           「飞机票怎么改签？」同理（改签 df=2）。

        第 2 条阈值取 0.5 是**量出来的**：32 道评测题里 24 道正例的最大缺失比只有 0.33
        （「藤上春有哪几种颜色？」里「藤上/春有」是分词噪声），所以 0.5 不误伤任何正例，
        同时挡住上面那两类域外提问。
        """
        if not content:
            return False
        if absent and not distinctive:
            return True
        return len(absent) / len(content) >= self.settings.absent_majority_ratio

    def _has_content_evidence(
        self,
        chunk: Chunk,
        content: set[str],
        distinctive: set[str],
        semantic_ok: bool,
    ) -> bool:
        """内容词证据，分三层：

        1. **基础层**：片段必须命中至少一个查询内容词。这一层就挡掉了「红烧肉怎么做才不腻？」
           那类纯语义噪声（相似度 0.5049 却没有一个查询词）。
        2. **生僻词层**：查询里有生僻词（且语料里确实存在）时，片段要么命中生僻词，
           要么**有语义证据**（向量相似度已达下限）。这一层挡掉「只命中『裙子』却排在
           藤上春前面」的尾巴，同时不冤枉那些确实在讲这件事、只是用词不同的片段。
        3. 查询全是常见词时，第 1 层即为全部要求。
        """
        if not content:
            return True
        tokens = self._chunk_content_tokens(chunk)
        if not (content & tokens):
            return False
        if not distinctive or semantic_ok:
            return True
        return bool(distinctive & tokens)

    def _has_keyword_evidence(self, chunk: Chunk, content: set[str]) -> bool:
        """关键词路在**没有**向量支持时独立送片段进结果的门槛，见模块常量说明。"""
        if not content:
            return True
        matched = len(content & self._chunk_content_tokens(chunk))
        return matched >= BM25_ONLY_MIN_CONTENT_MATCHES or matched == len(content)

    # ------------------------------------------------------------------ 融合
    def _route_relevance(self, name: str, score: float, top: float) -> float:
        """把某一路的原始分数换算成 [0, 1] 的相关度。"""
        if name == "vector":
            # 以 min_vector_score 为下界归一化：恰好等于阈值 → 0 分，完全相似 → 1 分
            floor = self.settings.min_vector_score
            span = max(1e-9, 1.0 - floor)
            return max(0.0, min(1.0, (score - floor) / span))
        return max(0.0, min(1.0, score / top)) if top else 0.0

    def _fuse_weighted(self, routes: Sequence[tuple[str, float, list[Retrieved]]]) -> dict[str, Retrieved]:
        fused: dict[str, Retrieved] = {}
        for name, weight, hits in routes:
            if not hits:
                continue
            top = max(hit.score for hit in hits) or 1.0
            for rank, hit in enumerate(hits, start=1):
                contribution = weight * self._route_relevance(name, hit.score, top)
                if contribution <= 0:
                    continue
                existing = fused.get(hit.chunk.chunk_id)
                if existing is None:
                    fused[hit.chunk.chunk_id] = Retrieved(
                        chunk=hit.chunk, score=contribution, source="hybrid", ranks={name: rank}
                    )
                else:
                    existing.score += contribution
                    existing.ranks[name] = rank
        return fused

    def _fuse_rrf(self, routes: Sequence[tuple[str, float, list[Retrieved]]]) -> dict[str, Retrieved]:
        k = self.settings.rrf_k
        fused: dict[str, Retrieved] = {}
        for name, weight, hits in routes:
            for rank, hit in enumerate(hits, start=1):
                contribution = weight / (k + rank)
                existing = fused.get(hit.chunk.chunk_id)
                if existing is None:
                    fused[hit.chunk.chunk_id] = Retrieved(
                        chunk=hit.chunk, score=contribution, source="hybrid", ranks={name: rank}
                    )
                else:
                    existing.score += contribution
                    existing.ranks[name] = rank
        return fused

    # ------------------------------------------------------------------ 对外
    def search(self, query: str, top_k: int | None = None, recall_k: int | None = None) -> list[Retrieved]:
        """两路召回 → 融合 → **两道闸门** → top_k。

        ## 为什么闸门要在融合**之后**再统一执行一遍

        最初两道闸门各自只作用于自己那一路（向量下限在向量路、内容词要求在 BM25 路），
        结果是**没有任何一路能独立证明相关性**，语料从 21 条涨到 690 条后立刻暴露：

        | 问题 | 漏进来的原因 |
        |---|---|
        | 「红烧肉怎么做才不腻？」 | 纯向量噪声：相似度 0.5049 > 下限，但片段里没有「红烧肉/不腻」 |
        | 「如何用 Docker 部署一个 Node 服务？」 | BM25 只命中一个疑问词「如何」（df=1） |
        | 「明天北京的天气怎么样？」 | BM25 只命中「北京」「明天」这类泛词，相似度仅 0.379 |

        所以现在改成**每一路的结果都必须同时满足**：

        * **向量绝对下限**：相似度 < ``min_vector_score`` 的片段一律不进结果，
          包括只被 BM25 命中的。代价是关键词路基本不再**引入**新片段，只对候选重新
          排序——但这是「能说出语料里没有」的前提，稠密检索永远会返回最近的若干条。
        * **内容词要求**：片段必须命中至少一个查询内容词。它就是防「语义上最近、
          话题上无关」的那一道。

        两个必要的例外，都是被真实数据逼出来的：

        * **关键词证据**：只被 BM25 命中的片段若命中 ≥2 个独立内容词，允许豁免向量
          下限（``BM25_ONLY_MIN_CONTENT_MATCHES``）。一刀切会把「谢安然今天穿的是
          哪个牌子？」这类**真答案**误拒——它的相似度低于下限，只靠关键词路才找得到。
        * **向量路整体失败**时（模型/向量库异常）降级为纯 BM25，不做向量下限判断：
          宁可召回宽一点，也不要整个检索不可用。
        """
        top_k = top_k or self.settings.top_k
        recall_k = recall_k or self.settings.recall_k
        floor = self.settings.min_vector_score

        content = BM25Retriever.content_tokens(tokenize(query))
        distinctive, absent = self._query_term_profile(content)

        # 用户问的核心概念语料里**根本没有**，剩下的又都是「lolita / lo」这类无处不在的词
        # （df=367/615）：此时任何一篇博文都能靠那个泛词匹配上，**连相似度都有 0.52 以上**
        # ——因为整个语料都是 lolita 内容，向量下限在这种「在域内但无答案」的提问上失效。
        # 唯一可靠的信号就是「核心概念不存在」，所以直接返回空，交给生成层说资料不足。
        # 反例：缺「亮片」（df=0）但还有「烫金」（df=1）可以锚定 —— 那种情况**不**拒答。
        if self._too_much_absent(content, absent, distinctive):
            logger.debug(
                "查询问到语料之外（缺失概念 %s，有区分度的词 %s），直接返回空",
                sorted(absent),
                sorted(distinctive),
            )
            return []

        # 向量路只调用一次：既用于本路召回，也用于判断其它路的命中是否有语义支持。
        vector_all: list[Retrieved] = []
        vector_ok = True
        # ⚠️「召回窗口」与「语义支持查询窗口」必须分开，这是踩出来的 bug：
        #   语料从 1642 涨到 1672 段后，「藤上春有哪几种颜色？」的标注帖仍是 BM25 第 1 名，
        #   却被新增片段挤出了向量 top-20 —— 它的相似度**查不到**、被当成 0，
        #   「语义支持」判定失败，整题从第 1 名变成返回空。
        #   它自己的相似度没变，变的是「还在不在窗口里」；而语料现在每晚都在涨，
        #   这种窗口边缘的误杀会反复出现。
        #   所以：参与**融合**的仍只有前 recall_k 条（不改变既有融合行为），
        #   但**语义支持**的查询用一个宽得多的窗口。
        support_k = max(recall_k, self.settings.semantic_support_k)
        try:
            vector_all = list(self.vector_search(query, support_k))
        except Exception as exc:  # noqa: BLE001 - 向量路挂了仍应能返回 BM25 结果
            vector_ok = False
            logger.warning("向量召回失败（%s），本次仅使用 BM25 结果", exc)
        vector_scores = {hit.chunk.chunk_id: hit.score for hit in vector_all}

        vector_hits = [hit for hit in vector_all[:recall_k] if hit.score >= floor]
        bm25_hits = list(self.bm25.search(query, recall_k))
        if vector_ok:
            bm25_hits = [
                hit
                for hit in bm25_hits
                if vector_scores.get(hit.chunk.chunk_id, 0.0) >= floor
                or self._has_keyword_evidence(hit.chunk, content)
            ]

        routes: list[tuple[str, float, list[Retrieved]]] = [
            ("bm25", self.settings.bm25_weight, bm25_hits),
            ("vector", self.settings.vector_weight, vector_hits),
        ]
        fused = self._fuse_rrf(routes) if self.settings.fusion_mode == "rrf" else self._fuse_weighted(routes)

        ordered = sorted(fused.values(), key=lambda hit: -hit.score)
        if content:
            ordered = [
                hit
                for hit in ordered
                if self._has_content_evidence(
                    hit.chunk,
                    content,
                    distinctive,
                    vector_scores.get(hit.chunk.chunk_id, 0.0) >= floor,
                )
            ]
        for hit in ordered:
            hit.source = "+".join(sorted(hit.ranks)) or "hybrid"
        # 精排放在**闸门之后**：它只重排已经通过判定的候选，因此拒答行为完全不变
        # （空仍是空），召回上限也不变——最差只是顺序不同。见 retriever/reranker.py。
        if self.reranker is not None and self.reranker.enabled:
            return self.reranker.rerank(query, ordered, top_k)
        return ordered[:top_k]
