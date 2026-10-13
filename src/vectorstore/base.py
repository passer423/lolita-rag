"""向量库接口定义。"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from src.schema import Chunk, Retrieved


class VectorStore(Protocol):
    dim: int

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int: ...

    def delete(self, chunk_ids: Sequence[str]) -> int: ...

    def search(self, query_vector: np.ndarray, top_k: int) -> list[Retrieved]: ...

    def count(self) -> int: ...

    def persist(self) -> None: ...

    def reset(self) -> None: ...
