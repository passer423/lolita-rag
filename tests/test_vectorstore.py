"""向量库测试：以可复现的简单向量验证存取、持久化与对齐逻辑。"""

from __future__ import annotations

import numpy as np
import pytest

from src.schema import Chunk
from src.vectorstore.simple_store import SimpleVectorStore


def make_chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id=f"c{i}", post_id=f"p{i}", text=f"文本{i}", url=f"https://example.com/{i}")
        for i in range(3)
    ]


def test_add_and_search_orders_by_cosine_similarity(tmp_path):
    store = SimpleVectorStore(tmp_path / "vectors.npz", dim=3)
    chunks = make_chunks()
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]], dtype=np.float32)
    store.add(chunks, matrix)

    hits = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)
    assert [hit.chunk.chunk_id for hit in hits] == ["c0", "c2"]
    assert hits[0].source == "vector"
    assert store.count() == 3


def test_persist_and_reload_binds_chunks_by_id(tmp_path):
    path = tmp_path / "vectors.npz"
    store = SimpleVectorStore(path, dim=2)
    chunks = make_chunks()
    store.add(chunks, np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32))
    store.persist()

    reloaded = SimpleVectorStore(path, dim=2).load()
    reloaded.bind_chunks(chunks)          # 向量与正文分开落盘，必须按 chunk_id 重新对齐
    assert reloaded.count() == 3
    hits = reloaded.search(np.array([0.0, 1.0], dtype=np.float32), top_k=1)
    assert hits[0].chunk.chunk_id == "c1"


def test_add_is_upsert_not_append(tmp_path):
    """正文变化的片段沿用了同一个 chunk_id，写入必须是覆盖而不是追加。

    这是增量重建最容易踩的坑：纯追加会让向量库比片段数多，检索时同一个
    来源出现两次（一条还是旧正文）。
    """
    store = SimpleVectorStore(tmp_path / "vectors.npz", dim=2)
    store.add(make_chunks(), np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32))
    assert store.count() == 3

    changed = Chunk(chunk_id="c0", post_id="p0", text="改过的文本", url="https://example.com/0")
    store.add([changed], np.array([[0.0, 1.0]], dtype=np.float32))

    assert store.count() == 3, "同 id 覆盖，条数不应增加"
    by_id = {hit.chunk.chunk_id: hit for hit in store.search(np.array([0.0, 1.0], dtype=np.float32), top_k=3)}
    assert by_id["c0"].chunk.text == "改过的文本"
    assert by_id["c0"].score == pytest.approx(1.0)


def test_delete_removes_vectors_and_chunks(tmp_path):
    store = SimpleVectorStore(tmp_path / "vectors.npz", dim=2)
    store.add(make_chunks(), np.eye(3, 2, dtype=np.float32))

    assert store.delete(["c1"]) == 1
    assert store.count() == 2
    assert store.delete(["不存在"]) == 0
    assert {hit.chunk.chunk_id for hit in store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)} == {"c0", "c2"}


def test_dimension_mismatch_is_rejected(tmp_path):
    store = SimpleVectorStore(tmp_path / "vectors.npz", dim=2)
    store.add(make_chunks()[:1], np.array([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="维度不一致"):
        store.add(make_chunks()[1:2], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))


def test_empty_store_returns_no_hits(tmp_path):
    store = SimpleVectorStore(tmp_path / "vectors.npz", dim=4)
    assert store.search(np.zeros(4, dtype=np.float32), top_k=5) == []
    assert store.count() == 0


def test_reset_clears_everything(tmp_path):
    path = tmp_path / "vectors.npz"
    store = SimpleVectorStore(path, dim=2)
    store.add(make_chunks(), np.eye(3, 2, dtype=np.float32))
    store.persist()
    store.reset()
    assert store.count() == 0
    assert not path.exists()
