"""纯 numpy 向量库。

存在的理由：本机 chromadb 的 rust 绑定出现过 ``os error 5``（权限/环境相关），
而这不应该是 RAG 流程的致命伤。这个实现零原生依赖，行为可预测，
用于 ``VECTOR_BACKEND=simple``，也是 chromadb 出错时的自动兜底。

语义上与 chroma 对齐：``add`` 是 upsert（按 ``chunk_id`` 覆盖），不是追加。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from src.schema import Chunk, Retrieved

logger = logging.getLogger(__name__)


class SimpleVectorStore:
    def __init__(self, path: Path, collection_name: str = "lolita_posts", dim: int = 0):
        self.path = Path(path)
        self.collection_name = collection_name
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray = np.zeros((0, dim or 0), dtype=np.float32)
        self._pending_ids: list[str] | None = None

    # ------------------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int:
        """写入向量，**按 chunk_id 覆盖**（upsert），语义与 chroma 对齐。

        必须是 upsert 而不是追加：增量重建时正文变化的片段沿用了同一个
        chunk_id，纯追加会让向量库里同时存在新旧两条，条数与片段数不再一致。
        """
        if len(chunks) == 0:
            return 0
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if not self._chunks:
            self._matrix = embeddings
            self._chunks = list(chunks)
            self.dim = int(self._matrix.shape[1])
            return len(chunks)

        if embeddings.shape[1] != self._matrix.shape[1]:
            raise ValueError(
                f"向量维度不一致：索引 {self._matrix.shape[1]}，输入 {embeddings.shape[1]}；"
                "换了 embedding 模型请先清空索引重建。"
            )

        positions = {chunk.chunk_id: i for i, chunk in enumerate(self._chunks)}
        rows = list(self._matrix)
        for chunk, vector in zip(chunks, embeddings):
            slot = positions.get(chunk.chunk_id)
            if slot is None:
                positions[chunk.chunk_id] = len(self._chunks)
                self._chunks.append(chunk)
                rows.append(vector)
            else:
                self._chunks[slot] = chunk
                rows[slot] = vector
        self._matrix = np.vstack(rows)
        self.dim = int(self._matrix.shape[1])
        return len(chunks)

    def delete(self, chunk_ids: Sequence[str]) -> int:
        """按 chunk_id 删除向量。增量重建时，正文变短会留下多余的旧片段。"""
        drop = set(chunk_ids)
        if not drop or not self._chunks:
            return 0
        keep = [i for i, chunk in enumerate(self._chunks) if chunk.chunk_id not in drop]
        removed = len(self._chunks) - len(keep)
        if not removed:
            return 0
        self._chunks = [self._chunks[i] for i in keep]
        self._matrix = (
            self._matrix[keep]
            if keep
            else np.zeros((0, self.dim or self._matrix.shape[1]), dtype=np.float32)
        )
        return removed

    def search(self, query_vector: np.ndarray, top_k: int) -> list[Retrieved]:
        if not self._chunks:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm
        scores = self._matrix @ query
        top_k = min(top_k, len(self._chunks))
        order = np.argsort(-scores)[:top_k]
        return [
            Retrieved(chunk=self._chunks[i], score=float(scores[i]), source="vector")
            for i in order
        ]

    def count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path,
            matrix=self._matrix,
            ids=np.array([chunk.chunk_id for chunk in self._chunks], dtype=object),
            dim=np.array([self.dim], dtype=np.int64),
        )

    def load(self) -> "SimpleVectorStore":
        if not self.path.exists():
            return self
        with np.load(self.path, allow_pickle=True) as data:
            self._matrix = data["matrix"].astype(np.float32)
            ids = [str(i) for i in data["ids"]]
            self.dim = int(data["dim"][0]) if "dim" in data else int(self._matrix.shape[1])
        self._pending_ids = ids
        return self

    def bind_chunks(self, chunks: Sequence[Chunk]) -> None:
        """把落盘的向量与 chunks.jsonl 里的正文按 chunk_id 对齐。"""
        mapping = {chunk.chunk_id: chunk for chunk in chunks}
        pending = getattr(self, "_pending_ids", None)
        if pending is None:
            return
        missing = [cid for cid in pending if cid not in mapping]
        if missing:
            logger.warning("有 %s 条向量找不到对应 chunk，索引可能已过期，建议重建", len(missing))
            keep = [i for i, cid in enumerate(pending) if cid in mapping]
            self._matrix = self._matrix[keep]
            pending = [pending[i] for i in keep]
        self._chunks = [mapping[cid] for cid in pending]
        self._pending_ids = None

    def reset(self) -> None:
        self._chunks = []
        self._matrix = np.zeros((0, self.dim or 0), dtype=np.float32)
        self.path.unlink(missing_ok=True)
