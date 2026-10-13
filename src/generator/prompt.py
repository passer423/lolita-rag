"""提示词构造。

两个硬约束写在系统提示里，且都有对应实现：
* 只允许依据检索资料回答（否则模型会拿训练记忆冒充语料事实）；
* 每个结论必须带 ``[编号]`` 引用，编号与 ``format_context`` 的顺序严格对应。
"""

from __future__ import annotations

from typing import Sequence

from src.schema import Retrieved

SYSTEM_PROMPT = """你是「Lolita 社区语料问答助手」。你的唯一知识来源是用户提供的【资料】。

必须遵守的规则：
1. 只能依据【资料】作答，禁止使用资料之外的任何信息，禁止编造。
2. 每一条结论后面都要标注来源编号，格式为 [1]、[2]；可以同时标注多个，如 [1][3]。
3. 如果【资料】不足以回答，直接说明「提供的资料中没有相关内容」，不要猜测、不要给通用建议。
4. 如果不同资料互相矛盾，要指出矛盾并分别标注来源，不要替用户做选择。
5. 用简体中文回答，先给结论，再给必要细节，不要复述问题、不要写开场白。
"""

NO_CONTEXT_REPLY = "没有检索到与该问题相关的资料，无法回答。请换一种问法，或先补充语料后重建索引。"


def format_context(hits: Sequence[Retrieved]) -> str:
    """把召回片段渲染成带编号的资料块。编号从 1 开始，与引用格式一致。"""
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        meta_parts: list[str] = []
        if chunk.author:
            meta_parts.append(f"作者：{chunk.author}")
        if chunk.created_at:
            meta_parts.append(f"时间：{chunk.created_at}")
        if chunk.total > 1:
            meta_parts.append(f"该博文第 {chunk.position + 1}/{chunk.total} 段")
        if chunk.url:
            meta_parts.append(f"链接：{chunk.url}")
        meta = f"\n（{'；'.join(meta_parts)}）" if meta_parts else ""
        blocks.append(f"[{index}] {chunk.text}{meta}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: Sequence[Retrieved]) -> list[dict[str, str]]:
    user_prompt = (
        f"【资料】\n{format_context(hits)}\n\n"
        f"【问题】\n{question.strip()}\n\n"
        "请依据上述资料回答，并按要求标注来源编号。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
