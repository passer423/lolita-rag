"""LLM-as-judge：检查答案**语义上**是否忠实于资料。

## 为什么必须有这一层

前面四道机器检查能抓「引用越界」「凭空数字」「无引用作答」「资料为空却硬答」，
但抓不到最常见也最隐蔽的一类错误——**语义不忠实**。实测过一个真实案例：

    问：宝石芭蕾的预约和尾款详情什么时候公布？
    答：…资料中仅提到宝石芭蕾将「提升各方面的效能与细节，做好交付」[2]，
        但未公布预约和尾款的具体时间或详情。

而标注博文正文里明写着「具体预约/尾款详情于 10.3-10.9 左右公布」，**且它就在上下文里**。
引用编号合法、没有编造数字、也不算硬答 —— 四道检查全部放行。这类「**资料里有、模型却说没有**」
以及「把资料的 A 说成 B」，只能靠另一个模型来读。

## 三条设计约束

1. **判定失败 ≠ 不忠实**：模型没吐 JSON、网络抖动、超时，都返回 ``faithful=None``
   （未知），绝不能算成「不忠实」——否则评审员自己的故障会变成系统的假阳性。
2. **不完整不算错**：答案只回答了一部分，只要说出来的都对，就算忠实。
   这一条必须写进提示词，否则评审员会把「答得少」判成「答得错」。
3. **要指出**具体哪句、错在哪**：只给一个 true/false 无法排查，也就无法改进提示词。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.config import Settings, settings as default_settings
from src.generator.prompt import format_context
from src.schema import Retrieved

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """你是严格的答案审校员。给定【资料】和【答案】，判断答案在**语义上**是否完全由资料支撑。

检查这些错误：
1. false_insufficiency：答案说「资料中没有X」，但资料里其实有 X（最常见、也最严重）；
2. misattribution：把资料里的 A 说成 B（张冠李戴、张三种的裙子算到李四头上）；
3. contradicted：答案与资料矛盾；
4. unsupported：答案里的结论在资料中找不到依据（不是指「资料没提」，而是答案自己下了资料没有的结论）；
5. wrong_citation：结论与其标注的编号对不上。

判分规则（务必遵守）：
- **不完整不算错**：答案只回答了一部分，只要说出来的都有依据，就是忠实；
- **引用编号本身不算事实**，只看内容；
- 只在**确凿**时才判不忠实；拿不准就判忠实。

只输出一个 JSON 对象，不要任何解释文字：
{"faithful": true 或 false, "issues": [{"kind": "上面五类之一", "detail": "具体哪句、错在哪"}]}"""


@dataclass(slots=True)
class JudgeIssue:
    kind: str
    detail: str


@dataclass(slots=True)
class JudgeVerdict:
    """一次评审结果。``faithful=None`` 表示**评审失败**（未知），不是「不忠实」。"""

    faithful: bool | None = None
    issues: list[JudgeIssue] = field(default_factory=list)
    error: str = ""
    raw: str = ""

    @property
    def failed(self) -> bool:
        return self.faithful is None


def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出 JSON 对象。

    不指望模型百分之百守格式（实测常见的多余包装是 ```json 代码块或前后加一句话），
    所以先整体试，再退化为「找第一个平衡的 {...}」。
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        payload = json.loads(candidate)
                        if isinstance(payload, dict):
                            return payload
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_verdict(text: str) -> JudgeVerdict:
    """把模型输出解析成判定；解析不出来就是「评审失败」，不是「不忠实」。"""
    payload = _extract_json(text)
    if payload is None:
        return JudgeVerdict(faithful=None, error="评审输出不是合法 JSON", raw=(text or "")[:400])

    faithful = payload.get("faithful")
    if not isinstance(faithful, bool):
        return JudgeVerdict(
            faithful=None, error="评审输出缺少 faithful 布尔字段", raw=(text or "")[:400]
        )

    issues: list[JudgeIssue] = []
    for item in payload.get("issues") or []:
        if isinstance(item, dict):
            issues.append(
                JudgeIssue(
                    kind=str(item.get("kind") or "unknown"),
                    detail=str(item.get("detail") or "").strip(),
                )
            )
        elif isinstance(item, str) and item.strip():
            issues.append(JudgeIssue(kind="unknown", detail=item.strip()))
    # 判「不忠实」却没给理由，等于没判——降级成评审失败，避免无从排查的假阳性
    if faithful is False and not issues:
        return JudgeVerdict(faithful=None, error="判不忠实但没给出理由", raw=(text or "")[:400])
    return JudgeVerdict(faithful=faithful, issues=issues, raw=(text or "")[:400])


class AnswerJudge:
    """用另一个模型读答案与资料，判断语义忠实度。"""

    def __init__(self, settings: Settings | None = None, client: Any | None = None):
        self.settings = settings or default_settings
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.llm_api_key.strip(),
                base_url=self.settings.llm_base_url.strip(),
                timeout=self.settings.llm_timeout,
            )
        return self._client

    def build_messages(self, question: str, answer_text: str, hits: Sequence[Retrieved]) -> list[dict]:
        user_prompt = (
            f"【资料】\n{format_context(hits)}\n\n"
            f"【问题】\n{question.strip()}\n\n"
            f"【答案】\n{answer_text.strip()}\n\n"
            "请判断【答案】是否完全由【资料】支撑，并按格式输出 JSON。"
        )
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def judge(self, question: str, answer_text: str, hits: Sequence[Retrieved]) -> JudgeVerdict:
        hits = list(hits)
        # 没有资料可依据时，语义忠实度无从谈起（那种情况由「检索为空必须拒答」管）
        if not hits:
            return JudgeVerdict(faithful=None, error="没有资料，跳过语义评审")
        if not self.settings.llm_enabled:
            return JudgeVerdict(faithful=None, error="未配置 LLM_API_KEY，无法评审")
        if not (answer_text or "").strip():
            return JudgeVerdict(faithful=None, error="答案为空，跳过语义评审")

        try:
            response = self._get_client().chat.completions.create(
                model=self.settings.llm_model,
                messages=self.build_messages(question, answer_text, hits),
                temperature=0.0,          # 评审要稳，不要发挥
                max_tokens=self.settings.llm_max_tokens,
            )
            text = (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - 评审失败不能影响主流程
            logger.warning("语义评审调用失败：%s", exc)
            return JudgeVerdict(faithful=None, error=f"评审调用失败：{exc}")

        verdict = parse_verdict(text)
        if verdict.failed:
            logger.warning("语义评审结果无法解析：%s", verdict.error)
        return verdict


def summarize_issues(verdicts: Sequence[JudgeVerdict], limit: int = 8) -> str:
    """把各条审批评语按类型汇总，便于一眼看出主要问题是什么。"""
    counter: dict[str, int] = {}
    for verdict in verdicts:
        for issue in verdict.issues:
            counter[issue.kind] = counter.get(issue.kind, 0) + 1
    if not counter:
        return "（没有发现问题）"
    ordered = sorted(counter.items(), key=lambda item: -item[1])
    return "、".join(f"{kind} ×{count}" for kind, count in ordered[:limit])
