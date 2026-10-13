"""运行时答案闸门：**不达标的模型答案不给用户**。

## 与 `quality.py` 的分工（这个区分很关键）

* `quality.py` 是**度量**：给评测用，必须看到模型**原始输出**，否则失败会被藏起来；
* 本模块是**拦截**：给服务用（`ask.py` / Web 接口），不达标就换成抽取式摘录。

所以校验**不能**塞进 `Generator.answer()`——那样 `eval_answer.py` 也拿不到原始输出，
评测就永远「全绿」了。

## 为什么降级是安全的

降级产物是**原文摘录**：每行「`[n]` + 该片段的原始正文」，一字不改。
它不可能编造引用编号或数字——因为那些内容就是语料本身。所以
**「模型编造」这件事从「可能返回给用户」变成「不可能返回给用户」**。

## 什么算「不达标」

只有 `quality.py` 里**机器可判定**的硬问题才触发降级：

* 引用编号越界（`[9]` 但只检索到 5 条）；
* 给了结论却没有任何引用；
* 答案里出现资料中没有的数字（疑似编造价格/日期）；
* 检索为空却没说明资料不足。

语义层面的不忠实（把资料的 A 说成 B）**抓不到**，那要 LLM-as-judge —— 文档里写清楚了，
不要因为它在本模块里缺席就以为有了保证。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import Settings, settings as default_settings
from src.generator.llm import Answer, Generator
from src.generator.quality import evaluate_answer
from src.schema import Retrieved

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VerifyOutcome:
    """一次运行时校验的结果，供日志与界面展示。"""

    checked: bool = False
    degraded: bool = False
    problems: list[str] = field(default_factory=list)
    note: str = ""


def verify_and_degrade(
    question: str,
    answer: Answer,
    hits: list[Retrieved],
    settings: Settings | None = None,
) -> tuple[Answer, VerifyOutcome]:
    """校验一条答案；有硬问题就换成抽取式摘录。

    返回 ``(答案, 校验结果)``。关掉开关（``ANSWER_VERIFY_ENABLED=false``）时原样返回，
    但 ``outcome.checked`` 会是 False——**别把「没检查」当成「检查通过」**。
    """
    settings = settings or default_settings
    if not settings.answer_verify_enabled:
        return answer, VerifyOutcome(checked=False)

    # 抽取式兜底本身就是原文照抄，没有校验的必要（也避免自我降级成同一个东西）
    if not answer.used_llm:
        return answer, VerifyOutcome(checked=True)

    quality = evaluate_answer(question, answer.text, hits, used_llm=True)
    if quality.ok:
        return answer, VerifyOutcome(checked=True)

    reason = "；".join(quality.problems)
    logger.warning("模型答案未通过运行时校验，已降级为原文摘录：%s", reason)
    degraded = Generator.extractive_answer(
        hits,
        note=f"模型答案未通过引用校验（{reason}），已降级为原文摘录",
    )
    return degraded, VerifyOutcome(
        checked=True,
        degraded=True,
        problems=list(quality.problems),
        note=reason,
    )
