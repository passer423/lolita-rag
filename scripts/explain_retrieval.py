"""逐路分解检索结果，用于回答「为什么返回了这些内容」。

它把 BM25 与向量的**原始分数**分别打印出来，而不是只给融合后的最终排序——
RRF 的 k=60 会刻意压平分数差距，光看融合分根本看不出某条结果是
「两路都真命中」还是「只被向量路凑数捞进来的」。

用法：
    python scripts/explain_retrieval.py "带亮片的裙子应该怎么洗？"
    python scripts/explain_retrieval.py "柄图" --top 8
    python scripts/explain_retrieval.py "茶会礼仪" --terms-only
    python scripts/explain_retrieval.py --scan          # 扫评测集，输出分数分布表（标定阈值用）
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
from _bootstrap import setup_logging

from src.config import settings
from src.pipeline.eval import DEFAULT_EVAL_PATH, load_eval_set
from src.pipeline.runtime import load_retriever
from src.retriever.bm25 import tokenize
from src.schema import Retrieved


def _fmt(hits: list[Retrieved], key: str) -> list[str]:
    lines = []
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        lines.append(
            f"    {rank:>2}. {hit.score:+.4f}  {chunk.chunk_id:<12} {chunk.text[:44]}"
        )
    return lines


def _scan(eval_path: str, top: int) -> int:
    """扫一遍评测集，把每题的分数分布打出来，用来标定阈值。

    判据：相关题目的最高向量相似度应当明显高于不相关题目；
    如果两者重叠，任何绝对阈值都会误伤。
    """
    bundle = load_retriever(settings)
    items = load_eval_set(eval_path)
    print(f"语料规模 : {len(bundle.chunks)} 个片段")
    print(f"评测集   : {eval_path}（{len(items)} 题）")
    print()
    header = f"{'类型':<8} {'向量top1':>9} {'向量#2':>8} {'BM25top1':>9} {'BM25top2':>9} {'比值':>6}  {'向量命中数(≥0.5/≥0.45/≥0.4)':<26} 问题"
    print(header)
    print("-" * len(header))

    for item in items:
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        label = "无语料" if item.get("expect_no_answer") else "相关"
        vec = bundle.retriever.vector_search(question, max(top, 3))
        bm = bundle.retriever.bm25.search(question, max(top, 3))
        v1 = vec[0].score if vec else 0.0
        v2 = vec[1].score if len(vec) > 1 else 0.0
        b1 = bm[0].score if bm else 0.0
        b2 = bm[1].score if len(bm) > 1 else 0.0
        ratio = (b2 / b1) if b1 else 0.0
        counts = "/".join(
            str(sum(1 for h in vec if h.score >= t)) for t in (0.5, 0.45, 0.4)
        )
        print(
            f"{label:<8} {v1:>9.4f} {v2:>8.4f} {b1:>9.4f} {b2:>9.4f} {ratio:>6.2f}  "
            f"{counts:<26} {question[:34]}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分解检索结果（BM25 / 向量 / 融合）")
    parser.add_argument("query", nargs="?", help="查询词或问题")
    parser.add_argument("--top", type=int, default=6, help="每路显示多少条（默认 6）")
    parser.add_argument("--terms-only", action="store_true", help="只看分词结果")
    parser.add_argument("--scan", action="store_true", help="扫描评测集输出分数分布表（标定阈值）")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_PATH), help="--scan 用的评测集")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    setup_logging(args.log_level or "WARNING")

    if args.scan:
        return _scan(args.eval_set, args.top)

    if not args.query:
        parser.error("需要给出 query，或使用 --scan")

    terms = tokenize(args.query)
    print(f"查询      : {args.query}")
    print(f"分词结果  : {terms}")
    if args.terms_only:
        return 0

    bundle = load_retriever(settings)
    total = len(bundle.chunks)
    print(f"语料规模  : {total} 个片段")
    print(
        f"当前参数  : RECALL_K={settings.recall_k}  TOP_K={settings.top_k}  "
        f"RRF_K={settings.rrf_k}  MIN_VECTOR_SCORE={settings.min_vector_score}"
    )
    print()

    bm25 = bundle.retriever.bm25
    vector_search = bundle.retriever.vector_search

    bm25_hits = bm25.search(args.query, args.top)
    vector_hits = vector_search(args.query, args.top)
    fused = bundle.retriever.search(args.query)

    print(f"BM25 原始分（score>0 才算命中，共按前 {args.top} 条显示）：")
    print("\n".join(_fmt(bm25_hits, "bm25")) or "    （无命中）")
    print()
    print(f"向量余弦相似度（前 {args.top} 条）：")
    print("\n".join(_fmt(vector_hits, "vector")) or "    （无命中）")
    print()

    if vector_hits:
        scores = [h.score for h in vector_hits]
        print(f"向量相似度区间：{min(scores):.4f} ~ {max(scores):.4f}"
              f"（极差 {max(scores) - min(scores):.4f}）")
    if bm25_hits:
        scores = [h.score for h in bm25_hits]
        print(f"BM25 分数区间  ：{min(scores):.4f} ~ {max(scores):.4f}"
              f"（极差 {max(scores) - min(scores):.4f}，相对最高分比值最低 {min(scores) / max(scores):.2f}）")
    print()

    print(f"融合后的最终结果（最多 TOP_K={settings.top_k} 条）：")
    for rank, hit in enumerate(fused, start=1):
        routes = "+".join(sorted(hit.ranks)) or "-"
        detail = " ".join(f"{name}#{pos}" for name, pos in sorted(hit.ranks.items()))
        print(
            f"    {rank:>2}. 融合分 {hit.score:.4f}  路径 {routes:<13} [{detail}]  "
            f"{hit.chunk.chunk_id:<12} {hit.chunk.text[:38]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
