"""构建/更新索引。

默认走**增量**：只对新增或正文变化的 chunk 计算向量，其余复用已有向量。
语料一旦开始增长，全量重建（每次几千条重新编码）会从「几秒」变成「几分钟」，
所以增量是 P1 的核心能力。

落盘产物：

    data/processed/chunks.jsonl   全部片段（正文）
    data/index/chroma/ 或 vectors.npz   向量
    data/index/manifest.json      元信息（后端 / 模型 / 维度 / 条数）
    data/cache/embeddings.pkl     按内容哈希缓存的向量

三种情况会**强制全量重建**：没有 manifest、embedding 模型变了、
向量库条数与 chunk 数不一致（索引被外部改动过）。这些都在返回值的 notes 里说明。
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

from src.config import PROJECT_ROOT, Settings, settings as default_settings
from src.crawler.mock import load_mock_posts
from src.embedding.embedder import BaseEmbedder, EmbeddingCache, get_embedder
from src.preprocessing.chunk import chunk_posts
from src.preprocessing.clean import clean_text, dedupe_posts
from src.pipeline.runtime import read_manifest, write_manifest
from src.schema import Chunk, Post
from src.storage import load_chunks, load_posts, save_chunks
from src.vectorstore import build_vector_store

logger = logging.getLogger(__name__)

ADD_BATCH = 256


@dataclass(slots=True)
class BuildStats:
    source: str
    mode: str = "incremental"          # incremental | rebuild
    posts: int = 0
    posts_after_dedupe: int = 0
    chunks: int = 0
    added: int = 0                     # 新增或正文变化的片段
    removed: int = 0                   # 已失效的片段（博文被删/改短/去重挤掉）
    unchanged: int = 0
    embedded: int = 0                  # 本次真正调用模型编码的条数
    backend: str = ""
    embedder: str = ""
    dim: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        mode_label = "全量重建" if self.mode == "rebuild" else "增量更新"
        lines = [
            f"数据来源      : {self.source}",
            f"更新模式      : {mode_label}",
            f"原始博文      : {self.posts} 条（去重后 {self.posts_after_dedupe} 条）",
            f"片段总数      : {self.chunks} 个（新增/变更 {self.added}，删除 {self.removed}，未变 {self.unchanged}）",
            f"本次编码      : {self.embedded} 条（其余复用向量缓存）",
            f"向量后端      : {self.backend}（{self.embedder}，{self.dim} 维）",
            f"耗时          : {self.seconds:.1f} 秒",
        ]
        if self.notes:
            lines.append("说明：")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


def _load_source_posts(source: str, settings: Settings) -> tuple[list[Post], str]:
    if source == "mock":
        return load_mock_posts(), "mock"
    if source == "raw":
        path = settings.raw_posts_path
        posts = load_posts(path)
        if not posts:
            raise FileNotFoundError(
                f"{path} 里没有数据。请先运行：python scripts/crawl.py --keyword Lolita"
            )
        # DATA_DIR 可以配到项目之外，此时 relative_to 会抛 ValueError
        try:
            label = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            label = str(path)
        return posts, label
    raise ValueError(f"未知的数据来源：{source}（可选 mock / raw）")


def build_index(
    source: str = "mock",
    settings: Settings | None = None,
    *,
    rebuild: bool = False,
    max_len: int = 400,
    overlap: int = 50,
    min_len: int | None = None,
    embedder: BaseEmbedder | None = None,
) -> BuildStats:
    settings = settings or default_settings
    settings.ensure_dirs()
    started = time.monotonic()
    notes: list[str] = []

    posts, source_label = _load_source_posts(source, settings)
    # 采集/加载之后统一再清洗一次，保证 mock 与真实数据走同一条路径
    for post in posts:
        post.text = clean_text(post.text)
    deduped = dedupe_posts(posts)
    # 短于 min_len 的博文不入库：它只有一个短语，构不成可检索的知识单元，
    # 却会在 BM25 里拿到很高的 idf 密度而挤掉真答案（依据见 settings.chunk_min_len）。
    chunks: list[Chunk] = chunk_posts(
        deduped,
        max_len=max_len,
        overlap=overlap,
        min_len=settings.chunk_min_len if min_len is None else min_len,
    )
    if not chunks:
        raise ValueError("清洗切分后没有任何有效片段，请检查语料内容或清洗规则")

    model = embedder or get_embedder(settings)
    manifest = read_manifest(settings)

    is_rebuild = rebuild
    if not is_rebuild and not manifest:
        is_rebuild = True
        notes.append("未找到 manifest，执行全量重建")
    elif not is_rebuild and manifest.get("embedder") != model.name:
        is_rebuild = True
        notes.append(
            f"embedding 模型从 {manifest.get('embedder')} 变为 {model.name}，执行全量重建"
        )

    existing: list[Chunk] = [] if is_rebuild else load_chunks(settings.chunks_path)
    backend = settings.vector_backend if is_rebuild else (manifest.get("backend") or settings.vector_backend)
    store, actual_backend = build_vector_store(settings, dim=model.dim, backend=backend)
    if actual_backend != backend:
        notes.append(f"向量后端 {backend} 不可用，已降级为 {actual_backend}")
        is_rebuild = True

    if not is_rebuild:
        bind = getattr(store, "bind_chunks", None)
        if callable(bind):
            bind(existing)
        if store.count() != len(existing):
            notes.append(
                f"向量库条数({store.count()})与现有片段数({len(existing)})不一致，执行全量重建"
            )
            is_rebuild = True

    if is_rebuild:
        store.reset()
        existing = []

    existing_by_id = {chunk.chunk_id: chunk for chunk in existing}
    new_ids = {chunk.chunk_id for chunk in chunks}
    to_add = [
        chunk
        for chunk in chunks
        if existing_by_id.get(chunk.chunk_id) is None
        or existing_by_id[chunk.chunk_id].text != chunk.text
    ]
    removed_ids = [chunk_id for chunk_id in existing_by_id if chunk_id not in new_ids]

    cache = EmbeddingCache(settings.embedding_cache_path, model)
    matrix = None
    if to_add:
        matrix = cache.encode([chunk.text for chunk in to_add], is_query=False)
        for start in range(0, len(to_add), ADD_BATCH):
            store.add(to_add[start : start + ADD_BATCH], matrix[start : start + ADD_BATCH])
        cache.save()
    if removed_ids:
        deleted = store.delete(removed_ids)
        logger.info("清理 %s 个已失效片段", deleted)
    store.persist()

    dim = int(
        model.dim
        or (matrix.shape[1] if matrix is not None else 0)
        or getattr(store, "dim", 0)
        or 0
    )
    save_chunks(settings.chunks_path, chunks)
    write_manifest(
        settings,
        source=source_label,
        mode="rebuild" if is_rebuild else "incremental",
        backend=actual_backend,
        embedder=model.name,
        dim=int(dim),
        chunks=len(chunks),
        posts=len(deduped),
    )

    stats = BuildStats(
        source=source_label,
        mode="rebuild" if is_rebuild else "incremental",
        posts=len(posts),
        posts_after_dedupe=len(deduped),
        chunks=len(chunks),
        added=len(to_add),
        removed=len(removed_ids),
        unchanged=len(chunks) - len(to_add),
        embedded=cache.last_misses,
        backend=actual_backend,
        embedder=model.name,
        dim=int(dim),
        seconds=time.monotonic() - started,
        notes=notes,
    )
    logger.info("索引更新完成：%s", stats.to_dict())
    return stats
