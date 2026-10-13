"""问答：混合检索 → 生成，并渲染成便于终端阅读的文本。"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from src.config import Settings, settings as default_settings
from src.generator.llm import Answer, Generator
from src.generator.verify import verify_and_degrade
from src.pipeline.query_log import log_query
from src.pipeline.runtime import RetrieverBundle, load_retriever
from src.schema import Retrieved

logger = logging.getLogger(__name__)


def ask(
    question: str,
    settings: Settings | None = None,
    *,
    top_k: int | None = None,
    bundle: RetrieverBundle | None = None,
    source: str = "cli",
) -> Answer:
    settings = settings or default_settings
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")

    bundle = bundle or load_retriever(settings)
    started = time.perf_counter()
    hits: list[Retrieved] = bundle.retriever.search(question, top_k=top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.debug("召回 %s 条：%s", len(hits), [(h.chunk.chunk_id, round(h.score, 4)) for h in hits])
    log_query(
        question, hits, settings=settings, source=source, elapsed_ms=elapsed_ms, bundle=bundle
    )
    # 运行时闸门：模型答案不达标就换成原文摘录（见 generator/verify.py）
    answer, outcome = verify_and_degrade(question, Generator(settings).answer(question, hits), hits, settings)
    if outcome.degraded:
        logger.warning("答案已降级为原文摘录：%s", outcome.note)
    return answer


def retrieve(
    question: str,
    settings: Settings | None = None,
    *,
    top_k: int | None = None,
    bundle: RetrieverBundle | None = None,
    source: str = "cli",
) -> list[Retrieved]:
    """只做**检索**，不调用生成层。

    检索和生成是两层：想先看看「语料里到底有没有相关内容」时，
    只看检索结果比看一段模型总结更有用——召回为空就说明语料里没有，
    召回一堆不相关的则说明该调阈值或重排序。

    ``source`` 会写进查询日志（``web`` / ``cli``）。**评测与排查脚本不要走这里**：
    它们直接调 ``bundle.retriever.search``，以免自己的评测污染真实查询统计。
    """
    settings = settings or default_settings
    question = (question or "").strip()
    if not question:
        raise ValueError("查询不能为空")
    bundle = bundle or load_retriever(settings)
    started = time.perf_counter()
    hits = bundle.retriever.search(question, top_k=top_k)
    log_query(
        question,
        hits,
        settings=settings,
        source=source,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        bundle=bundle,
    )
    return hits


def hits_to_dicts(hits: Sequence[Retrieved]) -> list[dict]:
    return [
        {
            "rank": index,
            "score": round(hit.score, 6),
            "retriever": hit.source,
            "post_id": hit.chunk.post_id,
            "chunk_id": hit.chunk.chunk_id,
            "author": hit.chunk.author,
            "created_at": hit.chunk.created_at,
            "url": hit.chunk.url,
            "text": hit.chunk.text,
        }
        for index, hit in enumerate(hits, start=1)
    ]


def render_hits(hits: Sequence[Retrieved], query: str = "") -> str:
    if not hits:
        return f"没有检索到与「{query}」相关的内容。"
    lines = [f"命中 {len(hits)} 条（按相关度从高到低）：", ""]
    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        where = " / ".join(part for part in (chunk.author, chunk.created_at) if part) or "未知来源"
        lines.append(
            f"[{index}] 相关度 {hit.score:.4f}｜召回路径 {hit.source}｜{where}"
        )
        if chunk.total > 1:
            lines.append(f"      （该博文第 {chunk.position + 1}/{chunk.total} 段）")
        lines.append(f"      {chunk.text}")
        if chunk.url:
            lines.append(f"      {chunk.url}")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_answer(answer: Answer) -> str:
    """终端输出：答案 + 来源清单 + 召回过程说明。"""
    lines = ["", answer.text, ""]
    if answer.hits:
        lines.append("来源：")
        for index, hit in enumerate(answer.hits, start=1):
            chunk = hit.chunk
            where = " / ".join(
                part for part in (chunk.author, chunk.created_at) if part
            ) or "未知来源"
            lines.append(
                f"  [{index}] {where}｜召回路径 {hit.source}｜相关度 {hit.score:.4f}"
            )
            if chunk.url:
                lines.append(f"      {chunk.url}")
    if answer.note:
        lines.append(f"（{answer.note}）")
    return "\n".join(lines)
