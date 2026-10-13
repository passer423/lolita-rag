"""切分策略。

微博是短文本：绝大多数博文一条就是一个检索单元，切碎反而会丢失上下文。
只有长文（长博文、长评、整理帖）才按句子边界开窗切分。
"""

from __future__ import annotations

import re

from src.preprocessing.clean import clean_text
from src.schema import Chunk, Post

DEFAULT_MAX_LEN = 400
DEFAULT_OVERLAP = 50
MIN_CHUNK_LEN = 8

# 中英文句末标点与换行都作为切分点，且保留标点本身
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]*[。！？!?；;\n]")


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.findall(text)
    tail = text[len("".join(parts)) :]
    if tail.strip():
        parts.append(tail)
    return [p for p in parts if p.strip()]


def chunk_text(text: str, max_len: int = DEFAULT_MAX_LEN, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """按句子边界开窗切分，窗口之间保留 ``overlap`` 个字符的重叠。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    buffer = ""
    for sentence in split_sentences(text):
        if not buffer:
            buffer = sentence
            continue
        if len(buffer) + len(sentence) <= max_len:
            buffer += sentence
            continue
        chunks.append(buffer)
        tail = buffer[-overlap:] if overlap > 0 else ""
        buffer = tail + sentence
    if buffer.strip():
        chunks.append(buffer)

    # 单句本身就超过 max_len 时，退化为定长切分
    result: list[str] = []
    step = max(1, max_len - overlap)
    for chunk in chunks:
        if len(chunk) <= int(max_len * 1.5):
            result.append(chunk)
        else:
            result.extend(chunk[i : i + max_len] for i in range(0, len(chunk), step))
    return [c.strip() for c in result if c.strip()]


def chunk_post(
    post: Post,
    max_len: int = DEFAULT_MAX_LEN,
    overlap: int = DEFAULT_OVERLAP,
    min_len: int = MIN_CHUNK_LEN,
) -> list[Chunk]:
    text = clean_text(post.text)
    if len(text) < min_len:
        return []
    pieces = chunk_text(text, max_len=max_len, overlap=overlap)
    total = len(pieces)
    return [
        Chunk(
            chunk_id=f"{post.id}#{index}",
            post_id=post.id,
            text=piece,
            url=post.url,
            author=post.author,
            created_at=post.created_at,
            likes=post.likes,
            position=index,
            total=total,
        )
        for index, piece in enumerate(pieces)
    ]


def chunk_posts(
    posts: list[Post],
    max_len: int = DEFAULT_MAX_LEN,
    overlap: int = DEFAULT_OVERLAP,
    min_len: int = MIN_CHUNK_LEN,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for post in posts:
        chunks.extend(chunk_post(post, max_len=max_len, overlap=overlap, min_len=min_len))
    return chunks
