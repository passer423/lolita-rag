"""cross-encoder 精排（reranker）。

## 为什么加它

混合检索（BM25 + 稠密向量，weighted 融合）是**双塔**式的：查询与片段各自编码，
再算相似度。它快，但**从不让查询和片段互相"看"**，所以对「同一话题但答非所问」
的片段分不开。cross-encoder 把「查询 + 片段」拼起来过一遍模型，判断力强得多。

实测指标（这一节数字由 `python scripts/rerank_eval.py` 产出，见 README）：
MRR 0.885 → 提升。它**只重排已经通过闸门的候选**，因此：

* 拒答行为完全不变（空就是空，重排不会把「没有资料」变成「有资料」）；
* 召回上限不变（Recall@k 不会因为它变差，最差也是打乱顺序）。

## 两个设计约束

1. **默认关闭**（`RERANKER_ENABLED=false`）。它要额外下载约 1GB 模型，且 CPU 上
   每个查询多花几百毫秒——这是使用者的取舍，不该由我替他决定。
2. **加载失败必须降级**，不能让整个检索挂掉：模型文件缺失、离线、内存不足都只记一条
   warning，然后原序返回。精排是**加分项**，不是依赖。
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

from src.config import Settings, settings as default_settings
from src.schema import Retrieved

logger = logging.getLogger(__name__)


def _sigmoid(value: float) -> float:
    """把 cross-encoder 的原始 logit 压到 0~1，便于与既有 score 语义对齐。"""
    try:
        return 1.0 / (1.0 + math.exp(-float(value)))
    except OverflowError:  # pragma: no cover - 极端 logit
        return 0.0 if value < 0 else 1.0


class Reranker:
    """对候选片段做 cross-encoder 精排。

    ``model`` 可注入（只要实现 ``predict(list[(query, text)]) -> list[float]``），
    因此测试不需要真的下载模型。
    """

    def __init__(self, settings: Settings | None = None, model: Any | None = None):
        self.settings = settings or default_settings
        self._model = model
        self._load_failed = False

    @property
    def enabled(self) -> bool:
        return bool(self.settings.reranker_enabled)

    def _get_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        try:
            import os

            if not self.settings.embedding_allow_download:
                # 与 embedder 同一套离线约定：默认不许联网，避免离线环境里长时间超时
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.settings.reranker_model)
            logger.info("精排模型已加载：%s", self.settings.reranker_model)
        except Exception as exc:  # noqa: BLE001 - 精排不可用不该让检索失败
            self._load_failed = True
            logger.warning("精排模型加载失败，本次跳过精排（检索仍可用）：%s", exc)
            return None
        return self._model

    def rerank(
        self, query: str, hits: Sequence[Retrieved], top_k: int
    ) -> list[Retrieved]:
        """重排候选并截断到 ``top_k``。

        关闭、模型不可用、或候选少于 2 条时都**原样返回**——
        精排是加分项，任何情况下都不该让结果变少或抛错。

        注意这里**不是**「候选不多于 top_k 就跳过」：即使候选恰好等于 top_k，
        顺序仍然影响 MRR，精排照样有价值。
        """
        candidates = list(hits)
        if not self.enabled or len(candidates) < 2:
            return candidates[:top_k]

        model = self._get_model()
        if model is None:
            return candidates[:top_k]

        # head 至少要够填满 top_k，同时按 RERANKER_TOP_N 多取一些参与打分——
        # 后者的价值就在这儿：融合排在第 15 名的片段可能是真正的答案，多打分才能把它捞上来。
        #
        # 不需要「尾巴」：head 长度 ≥ top_k，所以返回值永远来自打过分的候选。
        head = candidates[: max(top_k, self.settings.reranker_top_n)]
        pairs = [(query, hit.chunk.text) for hit in head]
        try:
            scores = list(model.predict(pairs, batch_size=self.settings.reranker_batch_size))
        except Exception as exc:  # noqa: BLE001 - 打分失败同样降级
            logger.warning("精排打分失败，保持原顺序：%s", exc)
            return candidates[:top_k]

        # 分数个数必须与候选一一对应；对不上说明模型返回异常，
        # 这时**宁可原序返回也不能丢候选**（zip 会静默截断，那是最难发现的 bug）。
        if len(scores) != len(head):
            logger.warning(
                "精排返回的分数个数（%s）与候选数（%s）不一致，保持原顺序", len(scores), len(head)
            )
            return candidates[:top_k]

        scored: list[Retrieved] = []
        for hit, raw in zip(head, scores):
            hit.score = _sigmoid(float(raw))
            hit.ranks["rerank"] = 0  # 占位：下面按新顺序写真实名次
            scored.append(hit)
        scored.sort(key=lambda item: -item.score)
        for index, hit in enumerate(scored):
            hit.ranks["rerank"] = index + 1
            hit.source = "+".join(sorted(hit.ranks)) or "hybrid"
        return scored[:top_k]
