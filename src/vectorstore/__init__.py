"""向量库工厂：按配置创建，并在原生依赖出问题时自动降级。"""

from __future__ import annotations

import logging

from src.config import Settings, settings as default_settings
from src.vectorstore.base import VectorStore
from src.vectorstore.simple_store import SimpleVectorStore

logger = logging.getLogger(__name__)

__all__ = ["VectorStore", "SimpleVectorStore", "build_vector_store"]


def build_vector_store(
    settings: Settings | None = None,
    *,
    dim: int = 0,
    backend: str | None = None,
) -> tuple[VectorStore, str]:
    """返回 ``(store, 实际生效的后端名)``。

    实际后端名需要写进 manifest：否则下次打开索引时无法知道该读 chroma 还是 npz。
    """
    settings = settings or default_settings
    backend = backend or settings.vector_backend

    if backend == "simple":
        store = SimpleVectorStore(settings.simple_vectors_path, settings.chroma_collection, dim)
        store.load()
        return store, "simple"

    try:
        from src.vectorstore.chroma_store import ChromaVectorStore

        return ChromaVectorStore(settings.chroma_path, settings.chroma_collection, dim), "chroma"
    except Exception as exc:  # noqa: BLE001 - chromadb 有原生依赖，失败必须可兜底
        logger.warning("chromadb 初始化失败（%s），已降级为 numpy 向量库", exc)
        store = SimpleVectorStore(settings.simple_vectors_path, settings.chroma_collection, dim)
        store.load()
        return store, "simple"
