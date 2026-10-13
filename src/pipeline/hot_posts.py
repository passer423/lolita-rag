"""热门帖子推荐（搜索页右侧那栏）。

## 两个词都得先定义清楚，否则功能就是骗人的

**「当日」按墙上时钟算是空的。** 实测本机 2026-10-13 跑的时候，语料里最新一条是
2026-10-12 20:26——今天一条都没有。所以：

* 锚点取**语料里最新一条的时间**，而不是 ``datetime.now()``；
* 界面上必须**如实标出是哪一天**（「语料最新 2026-10-12」），不能写「今日」；
* 同时把「语料截至什么时候」一并给出——语料是快照，用户有权知道它有多旧。

**「热门」不能只看点赞。** 实测语料里混着 25% 的域外帖（宽泛关键词搜进来的），
「最近 3 天按点赞排」的第一名是一条 11601 赞的时装周明星帖，与 Lolita 无关。
所以排序前先过一道**领域相关**过滤，并且：

* 匹配要**不区分大小写**——我自己第一版探测脚本就栽在这儿：词表里是 ``lolita``，
  而真实博文写的是 ``LOLITA``，于是把一堆真·圈内帖误判成域外帖；
* 同作者最多入选 N 条，否则一个品牌的转发抽奖能刷满整栏（实测确实会）。

打分用 ``likes + 2×comments + 3×reposts``：转发代表传播、评论代表讨论，
都比一个赞更有信息量。权重是产品口径，放在下面的常量里，要改就改这一处。
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from src.config import PROJECT_ROOT, Settings, settings as default_settings
from src.schema import Post

logger = logging.getLogger(__name__)

# 微博时间戳时区固定 +0800
TZ = timezone(timedelta(hours=8))

# 打分权重（产品口径，改这一处即可）
LIKES_WEIGHT = 1.0
COMMENTS_WEIGHT = 2.0
REPOSTS_WEIGHT = 3.0

# 少于这个字数的帖子不进热门栏（实测有一堆「OOTD」「ദ്ദി˶>ᴗo)」这种）
MIN_CHARS = 12

# 领域词表之外再补几个圈内常用说法。
# 为什么不直接往 jieba 用户词典里加：那会改变分词结果，进而影响已标定的检索指标
# （见 README 的评测章节）。热门栏只需要「是不是这条圈子的内容」，不该动分词。
EXTRA_DOMAIN_TERMS = (
    "lo娘", "lo汉", "三坑", "痛lo", "甜lo", "cla", "gothic", "sweet",
    "山正", "入坑", "退坑", "抽奖", "出物", "收物", "返图", "图透",
    "柄图", "裙撑", "罩衫", "内搭", "jsk", "op", "kc", "洛丽塔", "lolita",
)

# 形如 "Fri Oct 09 17:45:00 +0800 2026"
_WEIBO_TIME_RE = re.compile(
    r"([A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{2}:\d{2}:\d{2}) \+0800 (\d{4})"
)
# 形如 "2026-10-12 20:26:40" / "2026-10-12"
_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?")
# 形如 "10月11日 20:45"（HTML 导入路径的时间格式，**没有年份**）
_CN_RE = re.compile(r"(\d{1,2})月(\d{1,2})日\s*(\d{1,2}):(\d{2})")
_TAG_RE = re.compile(r"<[^>]+>")


def parse_post_time(text: str, *, default_year: int | None = None) -> datetime | None:
    """解析博文的 ``created_at``，失败返回 ``None``。

    三种格式对应三条采集路径：微博接口（带年份）、我们自己的 ISO 落盘、
    HTML 导入（``10月11日 20:45``，**缺年份**——只能靠 ``default_year`` 补，
    补不出来就返回 None，不猜）。
    """
    value = (text or "").strip()
    if not value:
        return None

    match = _WEIBO_TIME_RE.search(value)
    if match:
        try:
            return datetime.strptime(
                f"{match.group(2)} {match.group(1)}", "%Y %a %b %d %H:%M:%S"
            ).replace(tzinfo=TZ)
        except ValueError:
            return None

    match = _ISO_RE.match(value)
    if match:
        return datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4) or 0),
            int(match.group(5) or 0),
            int(match.group(6) or 0),
            tzinfo=TZ,
        )

    match = _CN_RE.match(value)
    if match and default_year:
        try:
            return datetime(
                default_year,
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
                tzinfo=TZ,
            )
        except ValueError:
            return None
    return None


def load_domain_terms(path: Path | None = None) -> tuple[str, ...]:
    """领域词表：jieba 用户词典 ∪ 补充词表（全部小写，匹配时不区分大小写）。"""
    terms: set[str] = set(EXTRA_DOMAIN_TERMS)
    dict_path = path or (PROJECT_ROOT / "config" / "jieba_userdict.txt")
    try:
        for line in dict_path.read_text(encoding="utf-8").splitlines():
            word = line.strip().split(" ")[0].strip()
            if word and not word.startswith("#"):
                terms.add(word)
    except OSError as exc:  # pragma: no cover - 词典缺失时退回补充词表
        logger.warning("读不到领域词表 %s：%s", dict_path, exc)
    return tuple(sorted(terms))


def is_domain_related(text: str, terms: Sequence[str]) -> bool:
    """正文是否出现领域词（**不区分大小写**）。

    大小写这一条是踩出来的：词表里写 ``lolita``，真实博文写 ``LOLITA``，
    不加 ``lower()`` 会把大量真·圈内帖误判成域外帖。

    匹配方式分两种（第二个坑也是踩出来的）：

    * **纯拉丁词**（``op`` / ``kc`` / ``cla`` / ``jsk``）按**词边界**匹配——
      否则 ``op`` 会命中 ``shop``、``cla`` 会命中 ``class``，实测把一条粉丝圈帖
      送进了 3 天窗口的第 2 名；
    * 其余词（``柄图``、``痛lo``、``lo娘``）按子串匹配。

    注意不能用 ``\\b``：在 Unicode 语义下中文字符也算 word 字符，
    ``\\bop\\b`` 匹配不到「的op是」。所以用负向环视。
    """
    return bool(matched_domain_term(text, terms))


def _term_pattern(terms: Sequence[str]) -> re.Pattern[str]:
    plain: list[str] = []
    bounded: list[str] = []
    for term in terms:
        lowered = term.lower()
        if not lowered:
            continue
        if re.fullmatch(r"[a-z0-9+#]+", lowered):
            bounded.append(re.escape(lowered))
        else:
            plain.append(re.escape(lowered))
    parts: list[str] = []
    if bounded:
        parts.append(r"(?<![a-z0-9])(?:" + "|".join(bounded) + r")(?![a-z0-9])")
    if plain:
        parts.append("(?:" + "|".join(plain) + ")")
    return re.compile("|".join(parts)) if parts else re.compile(r"(?!x)x")


def matched_domain_term(text: str, terms: Sequence[str]) -> str:
    """返回命中的第一个领域词（没命中返回空串）。

    为什么要有这个函数：排查「这条域外帖为什么混进来了」时，只知道「命中了」
    没有用，必须知道**命中了哪个词**——实测就是靠它发现 ``op`` 匹配到 ``shop``。
    """
    match = _term_pattern(terms).search((text or "").lower())
    return match.group(0) if match else ""


def clean_display_text(text: str, limit: int = 140) -> str:
    """给界面看的正文：去标签、解实体、压空白、截断。"""
    value = _TAG_RE.sub(" ", text or "")
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def hot_score(post: Post) -> float:
    """热门分：点赞 + 2×评论 + 3×转发。"""
    return (
        LIKES_WEIGHT * post.likes
        + COMMENTS_WEIGHT * post.comments
        + REPOSTS_WEIGHT * post.reposts
    )


@dataclass(slots=True)
class HotPost:
    post: Post
    when: datetime
    score: float
    text: str
    suggested_query: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.post.id,
            "author": self.post.author,
            "created_at": self.when.strftime("%Y-%m-%d %H:%M"),
            "url": self.post.url,
            "text": self.text,
            "likes": self.post.likes,
            "comments": self.post.comments,
            "reposts": self.post.reposts,
            "score": round(self.score, 1),
            # 界面上「用这条检索」按钮就搜这个词，并把词显示给用户（不藏着）
            "suggested_query": self.suggested_query,
        }


def suggest_query(
    text: str,
    document_frequency=None,
    *,
    domain_terms: Sequence[str] | None = None,
    fallback_chars: int = 18,
) -> str:
    """给一条热门帖挑一个「用这个搜」的词。

    ## 为什么优先挑领域词，而不是「最罕见的内容词」

    第一版就是按「语料里最罕见的内容词」挑的，结果实测挑出来的是
    「800」「plq」「穿刚」——数字、拼音缩写、分词碎片。罕见 ≠ 有主题性。

    现在的优先级：

    1. **正文里出现的领域词**（柄图 / jsk / 图透 / lo娘…），其中取语料里最罕见的那个；
    2. 没有领域词时，退到其他内容词，但**排除纯数字与短拉丁噪声**（``mm`` / ``plq``）；
    3. 都没有，就退回正文片段——至少点了有事发生。

    另外，``df == 0`` 的词一律不要：语料里根本没有的词当查询会被检索闸门直接拒答
    （实测过），点了按钮得到空结果比没有按钮更糟。
    """
    from src.retriever.bm25 import QUESTION_WORDS, STOPWORDS, tokenize

    terms = tuple(domain_terms) if domain_terms is not None else load_domain_terms()
    df = document_frequency if callable(document_frequency) else None

    def usable(token: str) -> bool:
        if len(token) < 2 or token in STOPWORDS or token in QUESTION_WORDS:
            return False
        if token.isdigit():
            return False
        if df is not None and df(token) <= 0:
            return False
        return True

    def rank(token: str) -> tuple[int, int]:
        return (int(df(token)) if df is not None else 1, -len(token))

    # 1) 领域词优先
    present = {match.group(0) for match in _term_pattern(terms).finditer((text or "").lower())}
    domain_hits = sorted((token for token in present if usable(token)), key=rank)
    if domain_hits:
        return domain_hits[0]

    # 2) 其他内容词，但挡掉纯数字与短拉丁噪声
    others = []
    for token in tokenize(text or ""):
        if not usable(token):
            continue
        if re.fullmatch(r"[a-z0-9+#]+", token) and len(token) < 3:
            continue
        others.append(token)
    if others:
        others.sort(key=rank)
        return others[0]

    # 3) 兜底：正文片段
    return clean_display_text(text, limit=fallback_chars + 1).rstrip("…")


@dataclass(slots=True)
class HotSelection:
    """热门栏的数据 + **它是怎么来的**（界面要如实展示，不能只给结果）。"""

    items: list[HotPost] = field(default_factory=list)
    window_days: int = 1
    anchor: datetime | None = None      # 窗口锚点 = 语料里最新一条的时间
    latest_post_at: datetime | None = None
    considered: int = 0                 # 窗口内候选数
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def anchor_label(self) -> str:
        return self.anchor.strftime("%Y-%m-%d") if self.anchor else ""

    def to_dict(self) -> dict:
        return {
            "items": [item.to_dict() for item in self.items],
            "window_days": self.window_days,
            "anchor": self.anchor.isoformat() if self.anchor else None,
            "anchor_label": self.anchor_label,
            "latest_post_at": self.latest_post_at.isoformat() if self.latest_post_at else None,
            "considered": self.considered,
            "skipped": dict(self.skipped),
        }


def select_hot_posts(
    posts: Iterable[Post],
    *,
    window_days: int = 1,
    limit: int = 8,
    per_author_cap: int = 2,
    min_chars: int = MIN_CHARS,
    domain_terms: Sequence[str] | None = None,
    require_domain: bool = True,
    document_frequency=None,
) -> HotSelection:
    """挑出「语料最新那几天」的热门帖。

    锚点用**语料里的最新时间**而不是墙上时钟——语料是快照，用它自己的时间轴才对得上；
    界面上再如实标出是哪一天（见模块文档）。

    窗口按**自然日**算：``days=1`` 是锚点当天 00:00 起，``days=3`` 是含锚点在内往前 3 天。
    """
    terms = tuple(domain_terms) if domain_terms is not None else load_domain_terms()
    selection = HotSelection(window_days=window_days)
    rows: list[Post] = list(posts)
    if not rows:
        return selection

    # 先确定年份兜底值（HTML 导入的时间戳缺年份），再解析全部时间
    default_year = None
    for post in rows:
        match = _WEIBO_TIME_RE.search(post.created_at or "") or _ISO_RE.match(post.created_at or "")
        if match:
            default_year = int(match.group(match.lastindex))
            break

    dated: list[tuple[Post, datetime]] = []
    for post in rows:
        when = parse_post_time(post.created_at, default_year=default_year)
        if when is not None:
            dated.append((post, when))
    if not dated:
        return selection

    anchor = max(when for _, when in dated)
    selection.anchor = anchor
    selection.latest_post_at = anchor
    # 按**自然日**算窗口，而不是「最近 24 小时」：界面上写的是「当日」，列表里就不该
    # 混进昨天晚上的帖子（实测第一版用滚动 24 小时，切换按钮说「当日」却混着前一天）。
    # days=1 → 锚点当天 00:00 起；days=3 → 含锚点在内往前数 3 个自然日。
    start_day = anchor.date() - timedelta(days=max(0, window_days - 1))
    cutoff = datetime.combine(start_day, time.min, tzinfo=TZ)

    skipped = {"window": 0, "domain": 0, "too_short": 0, "duplicate_author": 0}
    candidates: list[HotPost] = []
    for post, when in dated:
        if when < cutoff:
            skipped["window"] += 1
            continue
        text = clean_display_text(post.text)
        if len(text) < min_chars or not re.search(r"[\u4e00-\u9fff]", text):
            skipped["too_short"] += 1
            continue
        if require_domain and not is_domain_related(post.text, terms):
            skipped["domain"] += 1
            continue
        candidates.append(HotPost(post=post, when=when, score=hot_score(post), text=text))

    selection.considered = len(candidates)
    candidates.sort(key=lambda item: (-item.score, -item.when.timestamp()))

    # 同作者限量：实测一个品牌的转发抽奖能占掉整栏
    picked: list[HotPost] = []
    used: dict[str, int] = {}
    for item in candidates:
        author = item.post.author or "(未知)"
        if used.get(author, 0) >= per_author_cap:
            skipped["duplicate_author"] += 1
            continue
        used[author] = used.get(author, 0) + 1
        picked.append(item)
        if len(picked) >= limit:
            break

    # 建议查询放在最后算：只有入选的那几条需要它，没必要给所有候选都分词
    for item in picked:
        item.suggested_query = suggest_query(item.post.text, document_frequency)

    selection.items = picked
    selection.skipped = skipped
    return selection
