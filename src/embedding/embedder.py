"""文本向量化。

三种后端：
* ``bge``  —— sentence-transformers 加载 BGE 中文模型，效果最好（默认目标）；
* ``hash`` —— 纯 numpy 的字符 n-gram 哈希向量，确定性、零依赖、无需网络；
* ``auto`` —— 先试 BGE，失败（没装/没缓存/离线）自动降级到 hash 并给出告警。

降级机制的意义：索引和检索的接口完全一致，模型不可用时流程依然跑得通，
只是召回质量下降——这比直接崩掉更有用。
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from src.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)

HASH_DIM = 512


class BaseEmbedder:
    name: str = "base"
    dim: int = 0

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        raise NotImplementedError


class BGEEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        batch_size: int = 32,
        allow_download: bool = False,
        query_prefix: str = "",
    ):
        if not allow_download:
            # 必须先于 import/加载设置：否则 HuggingFace 会先尝试联网并长时间超时
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from sentence_transformers import SentenceTransformer  # 延迟导入，约 5-10 秒

        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.name = f"bge:{model_name}"
        self.model = SentenceTransformer(model_name, device="cpu")
        # sentence-transformers 6.x 把 get_sentence_embedding_dimension 改名了，
        # 这里两种都兼容，避免刷 FutureWarning
        get_dim = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension"
        )
        self.dim = int(get_dim())

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payload = [f"{self.query_prefix}{t}" for t in texts] if is_query and self.query_prefix else list(texts)
        vectors = self.model.encode(
            payload,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashEmbedder(BaseEmbedder):
    """字符 unigram + bigram 的带符号哈希向量（类似 hashing trick 的简化版）。

    对中文短文本足够给出「字面相似」的排序，作为 BGE 不可用时的兜底。
    用 blake2b 而非内置 hash()，保证跨进程、跨机器结果一致。
    """

    def __init__(self, dim: int = HASH_DIM):
        self.dim = dim
        self.name = f"hash:{dim}"

    @staticmethod
    def _features(text: str) -> list[str]:
        text = text.lower().strip()
        features: list[str] = []
        buffer: list[str] = []
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                if buffer:
                    features.append("".join(buffer))
                    buffer = []
                features.append(char)
            elif char.isalnum():
                buffer.append(char)
            else:
                if buffer:
                    features.append("".join(buffer))
                    buffer = []
        if buffer:
            features.append("".join(buffer))
        # 中文 bigram：补回一点词序信息
        cjk = [c for c in text if "\u4e00" <= c <= "\u9fff"]
        features.extend(a + b for a, b in zip(cjk, cjk[1:]))
        return features

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            vector[index] += 1.0 if (value >> 63) & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._encode_one(t) for t in texts]).astype(np.float32)


class EmbeddingCache:
    """按 (模型, 文本) 缓存向量；重建索引时绝大部分向量可直接复用。"""

    def __init__(self, path: Path, embedder: BaseEmbedder):
        self.path = Path(path)
        self.embedder = embedder
        self._store: dict[str, np.ndarray] = {}
        # 上一次 encode 真正调用了多少次模型（缓存未命中数）。
        # 增量重建要靠它来证明「只对新内容算了向量」。
        self.last_misses = 0
        if self.path.exists():
            try:
                with open(self.path, "rb") as handle:
                    self._store = pickle.load(handle)
            except Exception as exc:  # noqa: BLE001 - 缓存损坏不应阻断流程
                logger.warning("向量缓存读取失败（%s），将重新计算", exc)
                self._store = {}

    def _key(self, text: str, is_query: bool) -> str:
        raw = f"{self.embedder.name}|{'q' if is_query else 'd'}|{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.embedder.dim), dtype=np.float32)
        keys = [self._key(t, is_query) for t in texts]
        missing = [i for i, key in enumerate(keys) if key not in self._store]
        self.last_misses = len(missing)
        if missing:
            fresh = self.embedder.encode([texts[i] for i in missing], is_query=is_query)
            for slot, vector in zip(missing, fresh):
                self._store[keys[slot]] = vector
        return np.vstack([self._store[key] for key in keys]).astype(np.float32)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as handle:
            pickle.dump(self._store, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self) -> int:
        return len(self._store)


def get_embedder(settings: Settings | None = None) -> BaseEmbedder:
    settings = settings or default_settings
    backend = settings.embedding_backend

    if backend == "hash":
        return HashEmbedder()

    try:
        return BGEEmbedder(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            allow_download=settings.embedding_allow_download,
            query_prefix=settings.embedding_query_prefix,
        )
    except Exception as exc:  # noqa: BLE001 - 任何加载失败都应可降级
        if backend == "bge":
            raise
        logger.warning("BGE 加载失败（%s），已降级为 hash 向量后端，召回质量会下降", exc)
        return HashEmbedder()
