"""chromadb 后端。

两个刻意的设计：
1. **自己算向量**再交给 chroma，不用它的 embedding_function —— 避免 chroma
   自行下载内置 ONNX 模型（离线环境会卡住），也保证与检索端的模型一致；
2. 元数据整体序列化进单个 ``payload`` 字段 —— 规避 chroma 对元数据类型/空字符串的限制，
   目前不需要按元数据过滤。

备注：向量已做 L2 归一化，所以即使 hnsw 空间设置未生效（退化为 L2），
排序结果与余弦距离等价。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.schema import Chunk, Retrieved

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    def __init__(self, path: Path, collection_name: str = "lolita_posts", dim: int = 0):
        import chromadb

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._settings = chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
        self.client = chromadb.PersistentClient(path=str(self.path), settings=self._settings)
        self.collection = self._get_collection(collection_name)

    def _get_collection(self, name: str) -> Any:
        try:
            return self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # noqa: BLE001 - 不同版本对空间配置的写法不一致
            logger.debug("以 cosine 配置创建集合失败（%s），改用默认配置", exc)
            return self.client.get_or_create_collection(name=name)

    # ------------------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int:
        if len(chunks) == 0:
            return 0
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.dim and embeddings.shape[1] != self.dim:
            raise ValueError(f"向量维度不一致：索引 {self.dim}，输入 {embeddings.shape[1]}")
        self.dim = int(embeddings.shape[1])
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings.tolist(),
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"payload": json.dumps(chunk.to_dict(), ensure_ascii=False)} for chunk in chunks],
        )
        return len(chunks)

    def delete(self, chunk_ids: Sequence[str]) -> int:
        if not len(chunk_ids):
            return 0
        self.collection.delete(ids=list(chunk_ids))
        return len(chunk_ids)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[Retrieved]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=np.asarray(query_vector, dtype=np.float32).reshape(1, -1).tolist(),
            n_results=min(top_k, self.count()),
            include=["distances", "metadatas"],
        )
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[Retrieved] = []
        for metadata, distance in zip(metadatas, distances):
            chunk = Chunk.from_dict(json.loads(metadata["payload"]))
            hits.append(Retrieved(chunk=chunk, score=float(1.0 - distance), source="vector"))
        return hits

    def count(self) -> int:
        return int(self.collection.count())

    def persist(self) -> None:
        # PersistentClient 会自行落盘，这里无需额外动作
        return None

    def reset(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self._get_collection(name)
