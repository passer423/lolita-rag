"""生成层端到端评测：对评测集里的每个问题跑「检索 → 生成 → 质检」。

## 它和 `pipeline/eval.py` 的分工

* `pipeline/eval.py` —— 只看**检索**：Recall@k / MRR / 拒答（= 检索返回空）。
* `pipeline/answer_eval.py`（本模块）—— 看**用户真正看到的那段话**：
  引用编号是否有效、有没有结论不带引用、有没有编造资料外的数字、
  检索为空时有没有如实拒答。

两者会给出**不同的拒答数字**，这不是矛盾：前者问「找没找到资料」，
后者问「拿到资料后有没有老实说」。此前项目只有前者，所以 prompt 里那三条硬约束
一直没人验证。

## 没有 API Key 也能用

生成器是**注入**的（只要实现 ``answer(question, hits) -> Answer``）。所以：

* 没有 Key 时用真实的 ``Generator`` —— 它退化为抽取式答案，质检照样跑得通，
  能给出「兜底路径本身是否干净」的真实数字；
* 想验证质检器本身会不会抓错，注入假生成器即可（见 tests）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config import Settings, settings as default_settings
from src.generator.quality import AnswerQuality, evaluate_answer
from src.pipeline.runtime import RetrieverBundle, load_retriever
from src.schema import Retrieved

logger = logging.getLogger(__name__)


def load_questions(path: Path | str) -> list[dict]:
    """读评测集。``//`` 开头的注释行与空行跳过（与 ``pipeline/eval.py`` 同一约定）。"""
    items: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            items.append(json.loads(stripped))
        except json.JSONDecodeError:
            logger.warning("跳过无法解析的评测行：%s", stripped[:60])
    return items


@dataclass(slots=True)
class AnswerEvalReport:
    path: str = ""
    total: int = 0
    evaluated: int = 0
    used_llm: int = 0
    # 引用
    citation_valid: int = 0            # 引用编号全部有效的条数
    answers_without_citation: int = 0  # 给了结论但一条引用都没有
    uncited_sentences: int = 0
    sentences: int = 0
    # 事实
    answers_with_unsupported: int = 0
    unsupported_total: int = 0
    facts_checked: int = 0
    # 拒答
    no_context_cases: int = 0
    no_context_refused: int = 0
    with_context_refusals: int = 0
    # 明细
    results: list[AnswerQuality] = field(default_factory=list)

    @property
    def citation_valid_rate(self) -> float | None:
        return None if not self.evaluated else self.citation_valid / self.evaluated

    @property
    def unsupported_rate(self) -> float | None:
        return None if not self.evaluated else self.answers_with_unsupported / self.evaluated

    @property
    def refusal_rate_without_context(self) -> float | None:
        if not self.no_context_cases:
            return None
        return self.no_context_refused / self.no_context_cases

    @property
    def citation_coverage(self) -> float | None:
        if not self.sentences:
            return None
        return 1.0 - self.uncited_sentences / self.sentences

    @property
    def problems(self) -> list[AnswerQuality]:
        return [item for item in self.results if item.problems]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["citation_valid_rate"] = self.citation_valid_rate
        data["unsupported_rate"] = self.unsupported_rate
        data["refusal_rate_without_context"] = self.refusal_rate_without_context
        data["citation_coverage"] = self.citation_coverage
        data["problem_count"] = len(self.problems)
        # results 里已经含明细，这里不重复塞一份
        data.pop("results", None)
        return data

    def render(self) -> str:
        def percent(value: float | None) -> str:
            return "无用例" if value is None else f"{value:.3f}"

        lines = [
            f"评测集        : {self.path}",
            f"问题数        : {self.total}（已评测 {self.evaluated}，其中走 LLM {self.used_llm}）",
            "",
            f"引用编号有效  : {percent(self.citation_valid_rate)}（{self.citation_valid}/{self.evaluated}）",
            f"引用覆盖率    : {percent(self.citation_coverage)}"
            f"（{self.sentences - self.uncited_sentences}/{self.sentences} 句带引用）",
            f"无引用就作答  : {self.answers_without_citation} 条",
            f"疑似事实越界  : {percent(self.unsupported_rate)}"
            f"（{self.answers_with_unsupported}/{self.evaluated} 条，"
            f"涉及 {self.unsupported_total} 个数字 / 共查 {self.facts_checked} 个）",
            f"无资料时拒答  : {percent(self.refusal_rate_without_context)}"
            f"（{self.no_context_refused}/{self.no_context_cases}）",
            f"有资料也拒答  : {self.with_context_refusals} 条（不一定是错，但值得抽看）",
        ]
        problems = self.problems
        if problems:
            lines.append("")
            lines.append(f"有问题的答案 {len(problems)} 条：")
            for item in problems[:10]:
                lines.append(f"  · {item.question[:32]}：{'；'.join(item.problems)}")
            if len(problems) > 10:
                lines.append(f"  …… 另有 {len(problems) - 10} 条")
        return "\n".join(lines)


def evaluate_answers(
    eval_path: Path | str,
    settings: Settings | None = None,
    *,
    bundle: RetrieverBundle | None = None,
    generator=None,
    top_k: int | None = None,
    limit: int | None = None,
) -> AnswerEvalReport:
    """对评测集跑一遍「检索 → 生成 → 质检」。

    ``generator`` 可注入：只要实现 ``answer(question, hits) -> Answer``。
    不注入时用真实的 ``Generator``（无 Key 会自动退化为抽取式答案）。
    """
    settings = settings or default_settings
    if generator is None:
        from src.generator.llm import Generator

        generator = Generator(settings)

    items = load_questions(eval_path)
    if limit:
        items = items[:limit]

    bundle = bundle or load_retriever(settings)
    report = AnswerEvalReport(path=str(eval_path), total=len(items))

    for item in items:
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        # 评测直接调 retriever，**不经过 retrieve()**，因此不会写进查询日志
        hits: list[Retrieved] = bundle.retriever.search(question, top_k=top_k)
        answer = generator.answer(question, hits)
        quality = evaluate_answer(
            question, answer.text, hits, used_llm=bool(getattr(answer, "used_llm", False))
        )
        report.results.append(quality)
        report.evaluated += 1
        report.used_llm += int(quality.used_llm)
        report.citation_valid += int(quality.citation_valid)
        report.answers_without_citation += int(
            quality.has_context and not quality.refused and not quality.citations
        )
        report.uncited_sentences += quality.uncited_sentences
        report.sentences += quality.sentences
        report.answers_with_unsupported += int(bool(quality.unsupported))
        report.unsupported_total += len(quality.unsupported)
        report.facts_checked += quality.fact_total
        if not quality.has_context:
            report.no_context_cases += 1
            report.no_context_refused += int(quality.refused)
        elif quality.refused:
            report.with_context_refusals += 1

    return report
