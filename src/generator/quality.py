"""答案质量校验：机器可判定的那几项。

## 为什么要有这个模块

在此之前，项目只验证了**检索层**（Recall / MRR / 拒答=「检索返回空」），
而「生成层」——也就是用户真正看到的那段话——**从来没有被任何东西检查过**。
prompt 里写了三条硬约束（只依据资料、必须标 `[编号]`、资料不足要拒答），
但**没有任何断言在验证模型有没有照做**。

这个模块把其中**能机器判定**的部分固化下来：

| 检查 | 为什么它可判定 |
|---|---|
| 引用编号有效 | `[n]` 必须落在 `1..len(hits)` 内，越界就是硬错误 |
| 有结论就要有引用 | 正文里没有 `[n]` 却给了结论，说明约束没被遵守 |
| 数字/日期不能凭空出现 | 价格、日期、型号是这份语料里最要命的事实，逐个回查资料是否出现过 |
| 没资料时必须拒答 | 检索为空还给出结论 = 硬答 |

## 明确说清它**测不了**什么

* **语义层面的忠实**：把资料的 A 说成 B、把两篇的意思混起来，机器判不出来。
  这一项要 LLM-as-judge（本模块留了 ``judge`` 钩子），或者人工抽检。
* **单数字**：只提取 **≥2 位**的数字与小数。单个数字（「3 种颜色」）经常是模型对
  资料的正确归纳，把它判成编造属于误伤——宁可漏，不可冤。
* **换写法的数字**：资料写 `Time：10.20`、答案写 `10月20日`，靠「数字分量逐个回查」
  缓解（`10` 与 `20` 都在资料里即算支持），但格式差异很大时仍会误报。
  所以指标叫「**疑似**事实越界」，报告里会给出全部命中原文供人工复核。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from src.generator.prompt import format_context
from src.schema import Retrieved

# 引用编号：[1]、[ 12 ]、全角［2］
_CITATION_RE = re.compile(r"[\[［]\s*(\d+)\s*[\]］]")
_URL_RE = re.compile(r"https?://\S+")
# ≥2 位的数字（可带小数）：295、10.20、2026、10
_FACT_RE = re.compile(r"\d{2,}(?:\.\d+)?")
# 把正文切成句子，用于统计「有结论却没引用」的句子
_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?\n]?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 模型「承认资料不足」时常见的说法。命中任一即视为拒答。
_REFUSAL_MARKERS = (
    "资料中没有",
    "资料里没有",
    "没有相关的内容",
    "没有相关内容",
    "资料未提及",
    "资料没有提到",
    "没有提到",
    "无法回答",
    "无法确定",
    "无法给出",
    "资料不足",
    "没有足够",
    "未提供相关",
)


def normalize(text: str) -> str:
    """归一化：全角转半角、去掉空白。

    只做这几件事——归一化越激进，「事实在资料里出现过」的判断就越容易假阳性。
    """
    if not text:
        return ""
    out: list[str] = []
    for char in text:
        code = ord(char)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:      # 全角 ASCII → 半角
            out.append(chr(code - 0xFEE0))
        else:
            out.append(char)
    return re.sub(r"\s+", "", "".join(out)).lower()


def extract_citations(text: str) -> list[int]:
    """按出现顺序返回引用编号（去重）。"""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text or ""):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def is_refusal(text: str) -> bool:
    """模型是否在说「资料不足」。"""
    return any(marker in (text or "") for marker in _REFUSAL_MARKERS)


def extract_facts(text: str) -> list[str]:
    """提取需要回查资料的事实：≥2 位的数字与小数（引用编号、URL 先剔除）。"""
    cleaned = _URL_RE.sub(" ", text or "")
    cleaned = _CITATION_RE.sub(" ", cleaned)
    return _FACT_RE.findall(cleaned)


def unsupported_facts(answer_text: str, context: str) -> list[str]:
    """答案里出现、但资料里找不到的数字。**疑似**编造，需人工复核。

    判定用子串包含：资料里写 ``jsk=295``，答案写 ``295`` 也算出现。
    """
    haystack = normalize(context)
    return [fact for fact in extract_facts(answer_text) if normalize(fact) not in haystack]


@dataclass(slots=True)
class AnswerQuality:
    question: str
    has_context: bool
    n_sources: int
    refused: bool
    citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    fact_total: int = 0
    sentences: int = 0
    uncited_sentences: int = 0
    used_llm: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def citation_valid(self) -> bool:
        return not self.invalid_citations


def evaluate_answer(
    question: str,
    answer_text: str,
    hits: Sequence[Retrieved],
    *,
    used_llm: bool = False,
) -> AnswerQuality:
    """对一条答案跑全部机器可判定的检查。"""
    hits = list(hits)
    context = format_context(hits) if hits else ""
    citations = extract_citations(answer_text)
    refused = is_refusal(answer_text)
    invalid = [number for number in citations if number < 1 or number > len(hits)]
    unsupported = unsupported_facts(answer_text, context) if hits else []

    # 统计「有结论但没引用」的句子，粒度是**段落**。
    #
    # 为什么按段落而不是按句子：引用是按段落生效的，抽取式答案就是「[1] + 整段原文」，
    # 一段原文里往往有好几句话——按句子判定会把第 2 句起全算成「无引用」。
    #
    # 而「同一条引文内部含空行」的坑在**源头**解决了：抽取式生成会把引文里的空行
    # 折叠成空格，保证一条摘录就是一段。
    sentences = 0
    uncited = 0
    for paragraph in re.split(r"\n\s*\n", answer_text or ""):
        stripped = paragraph.strip()
        # 「以下是原文摘录」这类说明句不是结论
        if not stripped or stripped.startswith("（以下为检索到的原文摘录"):
            continue
        # 拒答句（「资料中没有相关内容」）本身没有结论，不该算进引用覆盖。
        # 实测漏掉这一条会把覆盖率误算成 0.930——那 16 句正是 8 条拒答 × 2 句。
        if is_refusal(stripped):
            continue
        if not _CJK_RE.search(stripped):
            continue
        paragraph_sentences = [
            item for item in _SENTENCE_RE.findall(paragraph) if item.strip() and _CJK_RE.search(item)
        ]
        if not paragraph_sentences:
            continue
        sentences += len(paragraph_sentences)
        if not _CITATION_RE.search(paragraph):
            uncited += len(paragraph_sentences)

    quality = AnswerQuality(
        question=question,
        has_context=bool(hits),
        n_sources=len(hits),
        refused=refused,
        citations=citations,
        invalid_citations=invalid,
        unsupported=unsupported,
        fact_total=len(extract_facts(answer_text)),
        sentences=sentences,
        uncited_sentences=uncited,
        used_llm=used_llm,
    )

    if invalid:
        quality.problems.append(f"引用编号越界：{invalid}（资料只有 {len(hits)} 条）")
    if not hits and not refused:
        quality.problems.append("检索为空却没有说明资料不足")
    if hits and not refused and not citations:
        quality.problems.append("给了结论却没有任何引用编号")
    if unsupported:
        quality.problems.append(f"疑似编造了资料里没有的数字：{unsupported}")
    return quality
