"""检索与拒答的离线评测。

指标选择理由：
* **Recall@k** —— 检索阶段最重要的指标：正确答案根本没被召回，后面生成再强也没用；
* **MRR** —— 关心「第一条相关结果排多靠前」，直接影响送进 prompt 的上下文质量；
* **拒答准确率** —— 语料里确实没有答案时，系统应该承认不知道而不是硬答。

评测集按 ``post_id`` 标注，与 chunk 切分方式解耦：改了切分策略不用重标。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config import PROJECT_ROOT, Settings, settings as default_settings
from src.pipeline.runtime import RetrieverBundle, load_retriever

logger = logging.getLogger(__name__)

DEFAULT_EVAL_PATH = PROJECT_ROOT / "eval" / "qa_mock.jsonl"


class EvalSetMismatchError(RuntimeError):
    """评测集里的标注 id 在当前语料里一个都不存在。

    典型场景：拿 ``eval/qa_mock.jsonl``（标注的是 mock-* 的 id）去打真实语料。
    此时 Recall 会算出 0.000，看起来像「检索彻底坏了」，实际是评测集和语料
    不是同一批数据。与其给出一个会被误读的数字，不如直接报错。
    """


@dataclass(slots=True)
class QuestionResult:
    question: str
    recall: float = 0.0
    reciprocal_rank: float = 0.0
    hit: bool = False
    refused: bool = False
    # 这一题**应该**拒答吗。渲染时必须区分「正例没召回」和「该拒却没拒」——
    # 实测踩过：只看 refused/hit 的话，拒答用例漏答会渲染成 [未命中]，
    # 和正例未命中长得一模一样，排查时根本分不清是哪种失败。
    expect_no_answer: bool = False
    retrieved: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvalReport:
    path: str = ""
    k: int = 0
    total: int = 0
    scored: int = 0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    hit_rate: float = 0.0
    refusal_total: int = 0
    refusal_correct: int = 0
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def refusal_accuracy(self) -> float | None:
        if self.refusal_total == 0:
            return None
        return self.refusal_correct / self.refusal_total

    def to_dict(self) -> dict:
        data = asdict(self)
        data["refusal_accuracy"] = self.refusal_accuracy
        return data

    def render(self) -> str:
        lines = [
            f"评测集        : {self.path}",
            f"问题数        : {self.total}（可计算召回 {self.scored} 条，拒答用例 {self.refusal_total} 条）",
            f"Recall@{self.k}     : {self.recall_at_k:.3f}",
            f"MRR           : {self.mrr:.3f}",
            f"Hit@{self.k}       : {self.hit_rate:.3f}",
        ]
        accuracy = self.refusal_accuracy
        lines.append(
            f"拒答准确率    : {'无相关用例' if accuracy is None else f'{accuracy:.3f}（{self.refusal_correct}/{self.refusal_total}）'}"
        )
        lines.append("")
        lines.append("逐题明细：")
        for index, result in enumerate(self.results, start=1):
            # 只用 GBK 也能编码的字符：Windows 中文控制台默认 cp936，
            # emoji 会让 print 直接抛 UnicodeEncodeError
            if result.expect_no_answer:
                # 拒答用例的两种结果要分得清：拒对了 / 该拒没拒
                flag = "[正确拒答]" if result.refused else f"[该拒未拒 {len(result.retrieved)} 条]"
            elif result.hit:
                flag = "[命中]"
            else:
                flag = "[未命中]"
            lines.append(
                f"  {flag} {index:>2}. recall={result.recall:.2f} rr={result.reciprocal_rank:.2f} "
                f"| {result.question}"
            )
        return "\n".join(lines)


def load_eval_set(path: Path | str | None = None) -> list[dict]:
    path = Path(path) if path else DEFAULT_EVAL_PATH
    if not path.exists():
        raise FileNotFoundError(f"评测集不存在：{path}")
    items: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
    return items


def _first_relevant_rank(retrieved: list[str], relevant: set[str]) -> int:
    for rank, post_id in enumerate(retrieved, start=1):
        if post_id in relevant:
            return rank
    return 0


def evaluate(
    path: Path | str | None = None,
    settings: Settings | None = None,
    *,
    top_k: int | None = None,
    bundle: RetrieverBundle | None = None,
) -> EvalReport:
    settings = settings or default_settings
    bundle = bundle or load_retriever(settings)
    k = top_k or settings.top_k
    items = load_eval_set(path)

    # ---- 先校验评测集与语料是否对应，避免输出误导性的 0.000
    known_post_ids = {chunk.post_id for chunk in bundle.chunks}
    labeled_ids = {
        str(post_id)
        for item in items
        for post_id in (item.get("relevant") or [])
    }
    if labeled_ids:
        missing = labeled_ids - known_post_ids
        if not (labeled_ids & known_post_ids):
            sample_missing = sorted(missing)[:3]
            sample_known = sorted(known_post_ids)[:3]
            raise EvalSetMismatchError(
                f"评测集 {path or DEFAULT_EVAL_PATH} 里标注的 post_id 在当前语料中一个都不存在"
                f"（标注示例：{sample_missing}；语料示例：{sample_known}）。"
                "这说明评测集与语料不是同一批数据——例如用 eval/qa_mock.jsonl 去打真实语料。"
                "请针对当前语料重新标注评测集（按 post_id 标注），再运行评测。"
            )
        if missing:
            logger.warning(
                "有 %s 个标注 post_id 不在当前语料中（共 %s 个），这些用例的召回会被低估；"
                "语料可能被裁剪过，或评测集标注了已删除的博文",
                len(missing),
                len(labeled_ids),
            )

    report = EvalReport(path=str(path or DEFAULT_EVAL_PATH), k=k, total=len(items))
    recall_sum = 0.0
    rr_sum = 0.0
    hit_count = 0

    for item in items:
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        hits = bundle.retriever.search(question, top_k=k)
        retrieved: list[str] = []
        for hit in hits:
            if hit.chunk.post_id not in retrieved:
                retrieved.append(hit.chunk.post_id)

        if item.get("expect_no_answer"):
            refused = len(hits) == 0
            report.refusal_total += 1
            report.refusal_correct += int(refused)
            report.results.append(
                QuestionResult(
                    question=question,
                    refused=refused,
                    expect_no_answer=True,
                    retrieved=retrieved,
                )
            )
            continue

        relevant = {str(x) for x in item.get("relevant", [])}
        if not relevant:
            continue
        found = relevant & set(retrieved)
        recall = len(found) / len(relevant)
        rank = _first_relevant_rank(retrieved, relevant)
        rr = 1.0 / rank if rank else 0.0
        report.scored += 1
        report.results.append(
            QuestionResult(
                question=question,
                recall=recall,
                reciprocal_rank=rr,
                hit=bool(found),
                retrieved=retrieved,
            )
        )
        recall_sum += recall
        rr_sum += rr
        hit_count += int(bool(found))

    if report.scored:
        report.recall_at_k = recall_sum / report.scored
        report.mrr = rr_sum / report.scored
        report.hit_rate = hit_count / report.scored
    return report
