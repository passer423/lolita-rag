"""落盘读写。

集中在一处，好处是索引文件格式变化时只需要改这一个文件；
所有写入都走「临时文件 + 原子替换」，避免中断时留下半截索引。
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Iterable, TypeVar

from src.schema import Chunk, Post

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    _atomic_write_text(path, payload + ("\n" if payload else ""))
    return payload.count("\n") + 1 if payload else 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("%s 第 %s 行不是合法 JSON，已跳过", path, line_no)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """追加一行。用于「只增不改」的流水记录（例如每日更新历史）。

    刻意不做原子替换：历史文件只会越写越长，整文件重写既慢又更容易在中断时损坏。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_pickle(path: Path) -> Any | None:
    if not Path(path).exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


# ------------------------------------------------------------------ 语义化封装
def save_posts(path: Path, posts: list[Post]) -> int:
    return write_jsonl(path, (post.to_dict() for post in posts))


def load_posts(path: Path) -> list[Post]:
    return [Post.from_dict(row) for row in read_jsonl(path)]


def save_chunks(path: Path, chunks: list[Chunk]) -> int:
    return write_jsonl(path, (chunk.to_dict() for chunk in chunks))


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.from_dict(row) for row in read_jsonl(path)]
