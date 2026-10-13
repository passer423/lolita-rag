"""文本清洗。

微博文本的噪声比正文还多：#话题#、@提及、短链、emoji、转发前缀、零宽字符。
这一步的输出直接决定检索质量，因此每一项清洗都有对应单元测试。
"""

from __future__ import annotations

import hashlib
import html
import re

from src.schema import Post

# --------------------------------------------------------------------- 正则
_TAG_RE = re.compile(r"<[^>]{1,200}>")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_SHORTLINK_RE = re.compile(r"(?:[a-zA-Z0-9-]+\.)?t\.cn/\S*", re.IGNORECASE)
_WEIBO_LINK_PLACEHOLDER_RE = re.compile(r"[Oo]?\s*网页链接")
_MENTION_RE = re.compile(r"@[\w\u4e00-\u9fff\-_.]{1,30}")
_TOPIC_RE = re.compile(r"#([^#\s]{1,40})#")
_REPOST_PREFIX_RE = re.compile(r"^(?:转发微博|轉發微博|Repost|repost)\s*[:：]?\s*")
_REPOST_CHAIN_RE = re.compile(r"//\s*@[\w\u4e00-\u9fff\-_.]{1,30}\s*[:：]?")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\u2060]")
# 只清理 emoji / 变体选择符 / 箭头符号，避开中西文标点（\u3000-\u303f、\uff00-\uffef）
_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"
    "\U00002600-\U000027bf"
    "\U00002b00-\U00002bff"
    "\U00002190-\U000021ff"
    "\U0000fe00-\U0000fe0f"
    "\U0001f1e6-\U0001f1ff"
    "\u200d\u2640\u2642"
    "]+"
)
_WHITESPACE_RE = re.compile(r"[ \t\u00a0\u3000]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{2,}")


def strip_html(text: str) -> str:
    return _TAG_RE.sub(" ", html.unescape(text or ""))


def normalize_width(text: str) -> str:
    """只归一化全角**字母/数字**与全角空格，刻意不动中文标点。

    不能用 ``unicodedata.normalize("NFKC", ...)``：它会把 ，：！？ 一并转成 ASCII，
    检索时无所谓（分词会丢掉标点），但把检索原文展示给用户时可读性明显变差——
    「尺码问题统一回复:官方尺码表」就是被 NFKC 改坏的样子。
    """
    chars: list[str] = []
    for char in text:
        code = ord(char)
        if (
            0xFF10 <= code <= 0xFF19      # ０-９
            or 0xFF21 <= code <= 0xFF3A   # Ａ-Ｚ
            or 0xFF41 <= code <= 0xFF5A   # ａ-ｚ
        ):
            chars.append(chr(code - 0xFEE0))
        elif code == 0x3000:              # 全角空格
            chars.append(" ")
        else:
            chars.append(char)
    return "".join(chars)


def strip_urls(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _SHORTLINK_RE.sub(" ", text)
    return _WEIBO_LINK_PLACEHOLDER_RE.sub(" ", text)


def strip_mentions(text: str) -> str:
    return _MENTION_RE.sub(" ", text)


def unwrap_topics(text: str) -> str:
    """``#Lolita#`` → ``Lolita``：话题词本身往往是有效检索信息，保留文字只去掉井号。"""
    return _TOPIC_RE.sub(r"\1", text)


def strip_repost_noise(text: str) -> str:
    text = _REPOST_PREFIX_RE.sub("", text)
    return _REPOST_CHAIN_RE.sub(" ", text)


def strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """清洗入口：顺序敏感，先结构后字符。"""
    if not text:
        return ""
    text = strip_html(text)
    text = strip_repost_noise(text)
    text = strip_urls(text)
    text = strip_mentions(text)
    text = unwrap_topics(text)
    text = strip_emoji(text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = normalize_width(text)
    return collapse_whitespace(text)


def content_hash(text: str) -> str:
    """稳定的内容指纹（跨进程一致，因此不能用内置 hash()）。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def dedupe_posts(posts: list[Post]) -> list[Post]:
    """按清洗后正文去重，保留点赞更高的那条（更可能是原始出处）。"""
    best: dict[str, Post] = {}
    for post in posts:
        key = content_hash(post.text)
        current = best.get(key)
        if current is None or post.likes > current.likes:
            best[key] = post
    return list(best.values())
