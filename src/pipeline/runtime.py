"""索引的装载与元信息管理，供 ask / eval 复用。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from src.config import Settings, settings as default_settings
from src.embedding.embedder import BaseEmbedder, EmbeddingCache, get_embedder
from src.retriever.bm25 import BM25Retriever
from src.retriever.hybrid import HybridRetriever
from src.schema import Chunk, Retrieved
from src.storage import load_chunks
from src.vectorstore import build_vector_store

logger = logging.getLogger(__name__)


class IndexMissingError(RuntimeError):
    """索引不存在或不完整，需要先构建。"""


def read_manifest(settings: Settings | None = None) -> dict:
    settings = settings or default_settings
    if not settings.manifest_path.exists():
        return {}
    try:
        return json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("manifest 解析失败，将按默认配置装载索引")
        return {}


def write_manifest(settings: Settings | None = None, **fields) -> dict:
    settings = settings or default_settings
    payload = {"built_at": datetime.now().isoformat(timespec="seconds"), **fields}
    settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def make_vector_search(cache: EmbeddingCache, store) -> Callable[[str, int], list[Retrieved]]:
    """把「查询 → 向量 → 向量库」封成一个可调用对象，喂给混合检索。"""

    def _search(query: str, top_k: int) -> list[Retrieved]:
        vector = cache.encode([query], is_query=True)[0]
        return store.search(vector, top_k)

    return _search


@dataclass(slots=True)
class RetrieverBundle:
    retriever: HybridRetriever
    chunks: list[Chunk]
    embedder: BaseEmbedder
    store: object
    backend: str
    manifest: dict


def load_retriever(settings: Settings | None = None) -> RetrieverBundle:
    settings = settings or default_settings
    chunks = load_chunks(settings.chunks_path)
    if not chunks:
        raise IndexMissingError(
            f"没有找到任何 chunk（{settings.chunks_path}）。请先运行：python scripts/build_index.py"
        )

    manifest = read_manifest(settings)
    embedder = get_embedder(settings)

    recorded = manifest.get("embedder")
    if recorded and recorded != embedder.name:
        logger.warning(
            "索引是用 %s 构建的，当前配置是 %s；维度不一致会直接报错，请重建索引",
            recorded,
            embedder.name,
        )

    backend = manifest.get("backend") or settings.vector_backend
    store, actual_backend = build_vector_store(settings, dim=embedder.dim, backend=backend)
    if actual_backend != backend:
        logger.warning("索引记录的后端是 %s，实际可用的是 %s", backend, actual_backend)

    bind = getattr(store, "bind_chunks", None)
    if callable(bind):
        bind(chunks)

    if store.count() == 0:
        logger.warning("向量库为空，本次检索将只使用 BM25 关键词召回")

    cache = EmbeddingCache(settings.embedding_cache_path, embedder)
    bm25 = BM25Retriever(chunks, min_ratio=settings.min_bm25_ratio)
    retriever = HybridRetriever(bm25, make_vector_search(cache, store), settings)
    return RetrieverBundle(
        retriever=retriever,
        chunks=chunks,
        embedder=embedder,
        store=store,
        backend=actual_backend,
        manifest=manifest,
    )
