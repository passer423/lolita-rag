"""答案生成。

关键设计：**没有 API Key 也必须能跑**。
未配置 ``LLM_API_KEY``（或调用失败）时退化为抽取式答案——直接把召回原文
按编号列出，并明确标注「未经模型归纳」。这样：
* 离线/无 Key 时流程可端到端验证；
* 出问题时能一眼看出是「检索坏了」还是「生成坏了」。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

from src.config import Settings, settings as default_settings
from src.generator.prompt import NO_CONTEXT_REPLY, build_messages
from src.schema import Retrieved

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Answer:
    text: str
    hits: list[Retrieved] = field(default_factory=list)
    used_llm: bool = False
    note: str = ""

    @property
    def citations(self) -> list[str]:
        """按引用编号顺序返回来源链接，供终端与界面展示。"""
        return [hit.chunk.url or hit.chunk.chunk_id for hit in self.hits]


class Generator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or default_settings
        self._client = None

    # ------------------------------------------------------------------ 内部
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.llm_api_key.strip(),
                base_url=self.settings.llm_base_url.strip(),
                timeout=self.settings.llm_timeout,
            )
        return self._client

    @staticmethod
    def _extractive(hits: Sequence[Retrieved], note: str) -> Answer:
        # 引文内部的空行要折叠掉：否则「一条摘录」会在输出里裂成两个段落，
        # 后一段看起来就没有引用了（实测把引用覆盖率从 1.000 误报成 0.930）。
        lines = [
            f"[{index}] {re.sub(r'\n+', ' ', hit.chunk.text.strip())}"
            for index, hit in enumerate(hits, start=1)
        ]
        text = "（以下为检索到的原文摘录，未经模型归纳）\n\n" + "\n\n".join(lines)
        return Answer(text=text, hits=list(hits), used_llm=False, note=note)

    # ------------------------------------------------------------------ 对外
    def answer(self, question: str, hits: Sequence[Retrieved]) -> Answer:
        if not hits:
            return Answer(text=NO_CONTEXT_REPLY, hits=[], used_llm=False, note="no_context")

        if not self.settings.llm_enabled:
            return self._extractive(hits, note="未配置 LLM_API_KEY，已退化为抽取式答案")

        try:
            response = self._get_client().chat.completions.create(
                model=self.settings.llm_model,
                messages=build_messages(question, hits),
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("模型返回空内容")
            return Answer(text=text, hits=list(hits), used_llm=True)
        except Exception as exc:  # noqa: BLE001 - 生成失败不应让整个问答崩掉
            logger.warning("LLM 调用失败（%s），已退化为抽取式答案", exc)
            return self._extractive(hits, note=f"LLM 调用失败：{exc}")
