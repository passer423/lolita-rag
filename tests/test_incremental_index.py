"""增量更新测试。

增量是 P1 的核心能力，因此这里用**可计数的假 embedder** 直接断言
「只对新内容调用了模型」，而不是只看最终条数——后者在写成全量重建时
也会通过，属于无效断言。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Settings
from src.embedding.embedder import BaseEmbedder, HashEmbedder
from src.pipeline.build_index import build_index
from src.schema import Post
from src.storage import load_chunks, save_posts
from src.vectorstore.simple_store import SimpleVectorStore


class CountingEmbedder(BaseEmbedder):
    """包一层计数，用来证明真正编码了几条。"""

    def __init__(self, inner: BaseEmbedder | None = None):
        self.inner = inner or HashEmbedder()
        self.name = self.inner.name
        self.dim = self.inner.dim
        self.encoded: list[str] = []

    def encode(self, texts, *, is_query: bool = False) -> np.ndarray:
        texts = list(texts)
        self.encoded.extend(texts)
        return self.inner.encode(texts, is_query=is_query)


def make_settings(tmp_path) -> Settings:
    # 用 numpy 向量库，避免单元测试依赖 chromadb 的原生实现
    return Settings(data_dir=tmp_path, vector_backend="simple", embedding_backend="hash")


def test_first_build_is_rebuild_and_second_is_noop(tmp_path):
    settings = make_settings(tmp_path)
    model = CountingEmbedder()

    first = build_index("mock", settings, embedder=model)
    assert first.mode == "rebuild"
    assert first.added == first.chunks
    assert first.embedded == first.chunks
    assert len(model.encoded) == first.chunks

    model.encoded.clear()
    second = build_index("mock", settings, embedder=model)
    assert second.mode == "incremental"
    assert second.added == 0
    assert second.removed == 0
    assert second.embedded == 0
    assert model.encoded == [], "内容没变时不应该再调用模型"
    assert second.chunks == first.chunks


def test_only_new_post_is_embedded(tmp_path):
    settings = make_settings(tmp_path)
    model = CountingEmbedder()
    base = [
        Post(id=f"p{i}", text=f"第{i}条测试内容，关于柄图、裙撑和茶会礼仪，写得足够长以便切成一个片段。")
        for i in range(3)
    ]
    save_posts(settings.raw_posts_path, base)

    first = build_index("raw", settings, embedder=model)
    assert first.chunks == 3
    assert first.embedded == 3

    model.encoded.clear()
    save_posts(
        settings.raw_posts_path,
        base + [Post(id="p99", text="新增的一条测试内容，讲的是痛lo 的日常化改造，同样足够长。")],
    )
    second = build_index("raw", settings, embedder=model)

    assert second.mode == "incremental"
    assert second.chunks == 4
    assert second.added == 1
    assert second.removed == 0
    assert second.unchanged == 3
    assert second.embedded == 1, "只有新增片段需要编码"
    assert len(model.encoded) == 1


def test_edited_post_is_reencoded_and_stale_chunks_removed(tmp_path):
    settings = make_settings(tmp_path)
    model = CountingEmbedder()
    long_text = "".join(f"第{i}句用来把这条博文撑到需要切分的长度。句号结尾。" for i in range(30))
    save_posts(settings.raw_posts_path, [Post(id="p1", text=long_text)])

    first = build_index("raw", settings, embedder=model, max_len=100, overlap=20)
    assert first.chunks > 2

    # 博文被改短：旧的 p1#1.. 片段必须被清理，否则索引里会留下幽灵内容
    # （改短后的正文仍要长于 chunk_min_len，否则整条博文不再入库——那是另一条测试）
    save_posts(settings.raw_posts_path, [Post(id="p1", text="改成很短的一句话，但依然长于片段下限，所以还能入库。")])
    second = build_index("raw", settings, embedder=model, max_len=100, overlap=20)

    assert second.chunks == 1
    assert second.added == 1
    assert second.removed == first.chunks - 1
    assert second.embedded == 1

    chunks = load_chunks(settings.chunks_path)
    assert [c.chunk_id for c in chunks] == ["p1#0"]
    store = SimpleVectorStore(settings.simple_vectors_path).load()
    store.bind_chunks(chunks)
    assert store.count() == 1, "向量库里不能残留已失效片段"


def test_changing_embedder_forces_rebuild(tmp_path):
    settings = make_settings(tmp_path)
    build_index("mock", settings, embedder=HashEmbedder(dim=512))
    stats = build_index("mock", settings, embedder=HashEmbedder(dim=64))

    assert stats.mode == "rebuild"
    assert stats.dim == 64
    assert any("全量重建" in note for note in stats.notes)


def test_explicit_rebuild_flag_readds_everything_but_reuses_cache(tmp_path):
    """全量重建会重新写入所有向量，但**不该重新调用模型**——缓存按内容哈希命中。

    这是刻意的设计：重建索引不该浪费算力，模型没换、文本没变，向量就是同一个。
    """
    settings = make_settings(tmp_path)
    model = CountingEmbedder()
    first = build_index("mock", settings, embedder=model)
    model.encoded.clear()

    stats = build_index("mock", settings, rebuild=True, embedder=model)
    assert stats.mode == "rebuild"
    assert stats.added == first.chunks, "重建应当把全部片段重新写入向量库"
    assert stats.embedded == 0, "缓存已命中，不应再调用模型"
    assert model.encoded == []


def test_store_count_matches_chunks_after_incremental(tmp_path):
    settings = make_settings(tmp_path)
    model = CountingEmbedder()
    save_posts(settings.raw_posts_path, [Post(id="a", text="第一条测试内容，足够长以形成一个片段。")])
    build_index("raw", settings, embedder=model)
    save_posts(
        settings.raw_posts_path,
        [
            Post(id="a", text="第一条测试内容，足够长以形成一个片段。"),
            Post(id="b", text="第二条测试内容，同样足够长以形成一个片段。"),
        ],
    )
    build_index("raw", settings, embedder=model)

    chunks = load_chunks(settings.chunks_path)
    assert len(chunks) == 2
    store = SimpleVectorStore(settings.simple_vectors_path).load()
    store.bind_chunks(chunks)
    assert store.count() == 2
    assert store.search(np.array([1.0] * store.dim, dtype=np.float32), top_k=2)


def test_empty_source_raises_file_not_found(tmp_path):
    settings = make_settings(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_index("raw", settings, embedder=HashEmbedder())
