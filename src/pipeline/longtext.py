"""长文补全：微博接口会把长博文截断，必须回去取全文。

## 问题是怎么发现的

排查「宝石芭蕾」那题时，我以为模型出现了「资料里有、却说没有」的语义不忠实——**结果是我错了**。
追下去才发现：那条博文在语料里只有 141 字，结尾是「… 具体 ... 全文」，
而它**真实长度是 1365 字**。也就是说，**模型说"资料里没有"是对的**。

## 规模（实测）

| | 数量 |
|---|---|
| 被截断的博文 | **331 / 1478（22.4%）** |
| 其中位数长度 | 151 字（未截断的是 69 字） |

被砍掉的恰恰是**最长、信息量最大**的那批（产品参数、活动细则、知识整理）。
它们全来自采集路径（`weibo` 源）——导入路径拿的是页面 HTML，是完整的。

## 为什么不能只把「全文」两个字删掉

删掉标记会让截断变得**不可见**：博文看起来完整，实际少了 90%。所以：

* 能取回全文 → 用全文替换（`https://m.weibo.cn/statuses/extend`）；
* 取不回来 → 把标记换成**显式的截断说明**，别装作完整。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from src.schema import Post

logger = logging.getLogger(__name__)

# 微博在截断处留下的标记：「... 全文」「…全文」「 全文」
_TRUNCATED_RE = re.compile(r"(?:\.{2,}|…|⋯)*\s*全文\s*$")
# 取不回全文时替换成的说明（让截断保持可见，而不是被悄悄抹掉）
TRUNCATION_NOTE = "（原文过长，被微博接口截断，未能取回全文）"


def is_truncated(text: str) -> bool:
    """这条正文是否被微博接口截断。"""
    return bool(_TRUNCATED_RE.search((text or "").rstrip()))


def strip_truncation_marker(text: str) -> str:
    """把截断标记换成显式说明（**不是**删掉就算完）。"""
    return _TRUNCATED_RE.sub(TRUNCATION_NOTE, (text or "").rstrip())


@dataclass(slots=True)
class RefreshStats:
    candidates: int = 0
    fetched: int = 0
    failed: int = 0
    skipped: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def updated(self) -> int:
        return self.fetched + self.failed


def refresh_long_posts(
    posts: Sequence[Post],
    fetch: Callable[[str], str],
    *,
    limit: int | None = None,
) -> tuple[list[Post], RefreshStats]:
    """把被截断的博文补成全文；取不回来就标注为「已截断」。

    ``fetch(post_id) -> str``：返回全文；失败请返回空串或抛异常（都会被记为 failed）。
    返回 ``(新的博文列表, 统计)``——不原地修改传入对象，便于测试与回滚。
    """
    stats = RefreshStats()
    result: list[Post] = []
    for post in posts:
        if not is_truncated(post.text):
            result.append(post)
            continue
        if limit is not None and stats.updated >= limit:
            stats.skipped += 1
            result.append(post)
            continue

        stats.candidates += 1
        try:
            full_text = (fetch(post.id) or "").strip()
        except Exception as exc:  # noqa: BLE001 - 单条失败不该中断整批补全
            logger.warning("取全文失败（%s）：%s", post.id, exc)
            full_text = ""

        if full_text and not is_truncated(full_text):
            clone = _clone(post, full_text)
            result.append(clone)
            stats.fetched += 1
            if len(stats.examples) < 5:
                stats.examples.append(f"{post.id}：{len(post.text)} → {len(full_text)} 字")
        else:
            # 取不回来也要处理：把「全文」两个字换成显式说明，避免假装完整
            result.append(_clone(post, strip_truncation_marker(post.text)))
            stats.failed += 1
    return result, stats


def _clone(post: Post, text: str) -> Post:
    payload = post.to_dict()
    payload["text"] = text
    return Post.from_dict(payload)


def summarize(stats: RefreshStats) -> str:
    lines = [
        f"待补全 {stats.candidates} 条：成功 {stats.fetched}，失败 {stats.failed}"
        + (f"，按上限跳过 {stats.skipped}" if stats.skipped else "")
    ]
    lines.extend(f"  · {item}" for item in stats.examples)
    return "\n".join(lines)


def candidates_in(posts: Iterable[Post]) -> list[Post]:
    return [post for post in posts if is_truncated(post.text)]
