"""从语料里挖「真实用户问过的查询」。

## 为什么要挖这个

第 (4) 项的目标是「用**真实查询**反过来精简检索闸门」，而评测集那 32 道题
是我读完语料手写的——**自己出题自己答题**，拿它做增删闸门的唯一依据不够。

语料本身就是一份现成的真实查询来源：微博上有大量 `【求助】…怎么…？`
`请问…哪件更…？` 这类帖子，**提问的人是真实用户，不是我**。它们分布的用词、
句式、信息量都和我的评测集不同，因此适合回答一个具体问题：
**关掉某条闸门后，真实提问的结果会不会变**。

## 它**不是**什么（必须说清）

* 这些不是「本系统真实收到的查询」——后者只有查询日志能给（`--from-log`），
  而日志需要真实使用才会积累；
* 它们**没有标注**，所以只能用于「行为变化」分析（结果集变没变），
  不能算 Recall/MRR。
* 挖出来的问题可能带口语噪声（「老婆萌」这类称呼），这正是真实分布的样子。

所以顺序是：`--from-log`（最真）> `--from-corpus`（次之，他人写的真实提问）>
评测集（我自己出的，最偏）。
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from src.schema import Chunk

# 强信号：出现这些就是「在求人 / 在问事」，不必再看标点
_REQUEST_RE = re.compile(
    r"(求助|请问|求问|想问|问一下|问下|有没有|有木有|能不能|可不可以|是不是|谁知道|蹲一个)"
)
# 弱信号：疑问词。中文口语经常不写问号（「怎么分辨山正啊」），所以要配合位置与否定式判断
_INTERROGATIVE_RE = re.compile(r"(怎么|怎样|如何|哪[些个条款种]|什么|多少)")
# 「怎么搭配都好看」「什么裙子都收」「不管怎么」这类是陈述，不是提问
_NEGATIVE_RE = re.compile(
    r"(怎么|怎样|如何|什么)[^，。！？]{0,6}(都|也)|不管怎么|无论怎么"
)
_QUESTION_RE = re.compile(r"[^。！？!?\n]*[？?]")
_TAIL_MARKER_RE = re.compile(r"(吗|呢|嘛)[。！!？?]")
# 开头的「【求助】」「#超话#」「@某人」「投稿：」这类壳，去掉后才是问题本身。
#
# ⚠️ 这里**只剥结构性包装**，不要往里塞具体措辞：我一度把某条博文的「递遮阳伞」
# 硬编码进来，那是拿一条样本过拟合。真实提问里的口语寒暄就让它留着——那正是真实分布。
# 壳之间要允许空格（实测「【投稿】 递遮阳伞，…」里就有一个空格）。
_PREFIX_RE = re.compile(
    r"^(?:(?:【[^】]{0,8}】|#[^#\s]{0,24}#|@[^\s:：]{1,20}|[【\[]?投稿[】\]]?[:：]?)\s*)+"
)

MIN_QUERY_LEN = 6
MAX_QUERY_LEN = 42
# 疑问词出现在开头这么多个字以内才算「在问」，避免正文中部顺带提到「怎么」被误判
HEAD_WINDOW = 24
# 短贴里出现疑问词基本就是在问（「怎么分辨山正啊 看不出来」），长贴才需要标点佐证
SHORT_POST_LEN = 40


def looks_like_question(text: str) -> bool:
    """这条博文是不是在**提问**（而不是在卖货、晒图、转发、感叹）。

    判定分强弱两档，都是被真实语料逼出来的：

    * **强信号**（求助/请问/有没有…）——出现即算，哪怕没写问号；
    * **弱信号**（怎么/如何/哪些…）——必须出现在开头 24 字内，且要么有问号/「吗呢嘛」，
      要么整条够短。中文口语常常全篇不写问号（实测「怎么分辨山正啊 看不出来 印刷倒是挺清晰」），
      只认标点会漏掉一大批真实提问；反过来「怎么搭配都好看」这种陈述要用否定式排掉。
    """
    text = (text or "").strip()
    if not text:
        return False
    if _REQUEST_RE.search(text[:HEAD_WINDOW]):
        return True
    if _NEGATIVE_RE.search(text):
        return False
    if not _INTERROGATIVE_RE.search(text[:HEAD_WINDOW]):
        return False
    has_mark = bool(_QUESTION_RE.search(text) or _TAIL_MARKER_RE.search(text))
    return has_mark or len(text) <= SHORT_POST_LEN


def extract_query(text: str, *, max_len: int = MAX_QUERY_LEN) -> str | None:
    """从一条博文里抽出「用户真正问出来的那句话」；抽不出好查询就返回 None。"""
    text = (text or "").strip()
    if not looks_like_question(text):
        return None

    match = _QUESTION_RE.search(text)
    raw = match.group(0) if match else text
    candidate = _PREFIX_RE.sub("", raw).strip(" \t，,、。；;：:")
    if not (MIN_QUERY_LEN <= len(candidate) <= max_len):
        return None
    # 剥壳之后要仍然是问句：否则可能把「【求助】出物…」这种剥成了陈述句
    if not (_REQUEST_RE.search(candidate) or _INTERROGATIVE_RE.search(candidate)):
        return None
    return candidate


def mine_queries(chunks: Sequence[Chunk], limit: int = 200) -> list[str]:
    """按语料顺序挖真实提问，去重后最多返回 ``limit`` 条。"""
    seen: set[str] = set()
    queries: list[str] = []
    for chunk in chunks:
        query = extract_query(chunk.text)
        if not query or query in seen:
            continue
        seen.add(query)
        queries.append(query)
        if len(queries) >= limit:
            break
    return queries


def summarize_queries(queries: Iterable[str], limit: int = 8) -> str:
    items = list(queries)
    if not items:
        return "（语料里没有挖到提问）"
    lines = [f"共 {len(items)} 条（示例）："]
    lines.extend(f"  · {item}" for item in items[:limit])
    return "\n".join(lines)
