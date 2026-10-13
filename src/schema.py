"""贯穿全流程的数据结构。

刻意用 dataclass 而不是 pydantic：这些对象在检索热路径上会被大量创建，
dataclass 的开销更低，而配置校验已经由 ``src.config`` 负责。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass(slots=True)
class Post:
    """一条微博原始博文。"""

    id: str
    text: str
    author: str = ""
    created_at: str = ""
    url: str = ""
    likes: int = 0
    comments: int = 0
    reposts: int = 0
    is_repost: bool = False
    source: str = "mock"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Post":
        return cls(**_pick(cls, data))


@dataclass(slots=True)
class Chunk:
    """检索的最小单元。短博文一条就是一个 chunk，长文按窗口切分。"""

    chunk_id: str
    post_id: str
    text: str
    url: str = ""
    author: str = ""
    created_at: str = ""
    likes: int = 0
    position: int = 0          # 该 chunk 在原博文中的序号，从 0 开始
    total: int = 1             # 原博文被切成几段

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**_pick(cls, data))


@dataclass(slots=True)
class Retrieved:
    """一路召回的结果，``score`` 越大越相关。"""

    chunk: Chunk
    score: float = 0.0
    source: str = ""                                  # bm25 / vector / hybrid
    ranks: dict[str, int] = field(default_factory=dict)


def _pick(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """只保留 dataclass 认识的字段，兼容旧版本落盘的数据。"""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}
