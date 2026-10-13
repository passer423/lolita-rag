"""BM25 关键词检索。

中文必须先分词，且要处理停用词，否则「的」「了」这类高频字会把 idf 拉平。
专有词（JSK、OP、柄图、痛lo）正是 BM25 相对稠密向量的优势区间，
所以混合检索里这一路不能省。
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import jieba
from rank_bm25 import BM25Okapi

from src.config import settings as default_settings
from src.schema import Chunk, Retrieved

logger = logging.getLogger(__name__)
jieba.setLogLevel(logging.WARNING)

STOPWORDS: frozenset[str] = frozenset(
    """
    的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好
    自己 这 那 他 她 它 们 个 中 为 与 及 或 而 但 如果 因为 所以 可以 这个 那个 什么
    怎么 为什么 还是 就是 已经 一下 一些 有点 比较 非常 真的 其实 然后 而且 不过 只是
    吧 呢 啊 吗 呀 哦 哈 嗯 啦 嘛 之 于 以 被 把 让 从 对 后 前 里 外 时 都 又 再 更 最
    这条 那条 这种 那种 这款 那款 这次 那次 这些 那些 这边 那边 这里 那里 本次 该
    """.split()
)

# 疑问词与泛化提问词：它们**看起来**像内容词（长度 ≥2），其实是疑问结构的一部分，
# 任何文档都可能碰巧命中。实测「如何用 Docker 部署一个 Node 服务？」的 BM25 头名
# 是一篇日语系的投稿，全靠一个「如何」命中（df=1，idf 极高）——去掉它，关键词路
# 立刻变空，系统就会正确地拒答。
# 要求：只收**疑问/泛指**词，不收「怎么做」里的实义动词，否则会误伤真实提问。
QUESTION_WORDS: frozenset[str] = frozenset(
    """
    如何 怎么样 怎样 怎么 哪些 哪个 哪种 哪家 哪儿 哪里 几种 几个 是否 有没有 多少
    何时 为何 什么 为什么 请问 谁知道 求问 想知道
    哪几种 哪几款 哪条 哪件 哪款 几条 几款 哪些款
    怎么办 咋办 怎样办 如何办 是什么 有哪些 有什么 为什么 有没有
    """.split()
)

_CJK_RANGE = ("\u4e00", "\u9fff")
_USERDICT_LOADED = False

# 「A 字撑」这种「单个拉丁字母 + 空格 + 中文」的写法在本圈很常见（还有 OP 款、JSK 裙）。
# 空格会让自定义词典里的 a字撑 永远匹配不上，于是被切成「字撑」这种碎片。
# 只删掉字母/数字与中文之间的空格，不动中文之间的空格（那通常是并列关系）。
_LATIN_CJK_SPACE = re.compile(r"([A-Za-z0-9])[ \t\u3000]+(?=[\u4e00-\u9fff])")


def load_userdict(path: Path | str | None = None) -> bool:
    """加载 Lolita 圈的领域词典，只生效一次。

    没有它，``柄图`` 会被切成 ``柄`` + ``图`` 两个字——而 ``图`` 在
    「官图/图片/截图」里到处都是，单字会显著稀释 BM25 的区分度，
    前端高亮也会满屏都是黄底。改词典不需要重建索引：
    BM25 的语料在装载索引时重新分词，向量部分不受影响。
    """
    global _USERDICT_LOADED
    if _USERDICT_LOADED:
        return True
    _USERDICT_LOADED = True

    path = Path(path) if path else Path(default_settings.jieba_userdict)
    if not path.exists():
        logger.warning("自定义词典不存在，社区专有词会被切碎：%s", path)
        return False
    try:
        jieba.load_userdict(str(path))
    except Exception as exc:  # noqa: BLE001 - 词典坏了不该让检索直接不可用
        logger.warning("自定义词典加载失败（%s）：%s", path, exc)
        return False
    logger.info("已加载自定义词典：%s", path)
    return True


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    normalized = _LATIN_CJK_SPACE.sub(r"\1", (text or "").lower())
    for raw in jieba.lcut(normalized):
        token = raw.strip()
        if not token or token in STOPWORDS or token in QUESTION_WORDS:
            continue
        if len(token) == 1 and not (_CJK_RANGE[0] <= token <= _CJK_RANGE[1]):
            continue  # 丢掉单个标点或单个字母
        tokens.append(token)
    return tokens


load_userdict()


class BM25Retriever:
    """语料规模不大时直接在内存构建，索引的唯一持久化形式就是 chunks.jsonl。

    内置两道闸门，都是为了剪掉「匹配上了但没意义」的结果：

    * **内容词要求**：查询里若存在长度 ≥2 的词，文档必须至少命中其中一个。
      否则单个汉字就能造成假命中——实测「怎么用 Python 写快速排序？」
      会命中一篇讲缎带手作的博文，只因里面有个「用」字。
    * **相对分数下限**：低于「本查询最高分 × ``min_ratio``」的命中丢弃。
      BM25 绝对值随语料规模变化，所以只能用相对值。
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        min_ratio: float = 0.0,
    ):
        self.chunks: list[Chunk] = list(chunks)
        self.corpus_tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.token_sets = [set(tokens) for tokens in self.corpus_tokens]
        # 文档频率：某个词出现在多少个片段里。融合层用它判断查询词有没有区分度
        # （见 hybrid.py：查询里有生僻词时，片段必须命中生僻词）。
        self.document_frequencies: Counter[str] = Counter()
        for token_set in self.token_sets:
            self.document_frequencies.update(token_set)
        self.min_ratio = min_ratio
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=k1, b=b) if self.chunks else None

    def document_frequency(self, token: str) -> int:
        """该词出现在多少个片段里；语料里没出现过就是 0。"""
        return int(self.document_frequencies.get(token, 0))

    @property
    def corpus_size(self) -> int:
        return len(self.chunks)

    @staticmethod
    def content_tokens(query_tokens: Sequence[str]) -> set[str]:
        """长度 ≥2 的词才算「内容词」。单字中文太常见，不足以证明相关。"""
        return {token for token in query_tokens if len(token) >= 2}

    def search(self, query: str, top_k: int = 20) -> list[Retrieved]:
        if not self.chunks or self.bm25 is None:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # 注意：BM25Okapi 的 idf = log((N-df+0.5)/(df+0.5))，当某个词出现在
        # 「语料规模的一半」左右时 idf 会退化为 0。文档数只有 1~2 篇时几乎所有词
        # 都满足这一点，BM25 分数会全是 0——此时本方法返回空，检索只能靠向量路。
        # 语料 ≥4 篇就有正常信号了，所以这只是极小球场的边界，不是 bug。
        scores = self.bm25.get_scores(query_tokens)
        best = float(max(scores))
        if best <= 0:
            return []
        floor = best * self.min_ratio
        content = self.content_tokens(query_tokens)

        order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
        hits: list[Retrieved] = []
        for index in order:
            score = float(scores[index])
            # order 是降序的，一旦低于下限，后面的只会更小
            if score <= 0 or score < floor:
                break
            if content and not (content & self.token_sets[index]):
                continue
            hits.append(Retrieved(chunk=self.chunks[index], score=score, source="bm25"))
            if len(hits) >= top_k:
                break
        return hits

    def __len__(self) -> int:
        return len(self.chunks)
