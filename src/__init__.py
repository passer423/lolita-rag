"""lolita-rag：微博 Lolita 社区语料的检索增强问答（RAG）工具包。

模块划分对应数据流水线的各个阶段：

    crawler       采集（真实微博 / mock 语料）
    preprocessing 清洗与切分
    embedding     文本向量化（BGE，可降级）
    vectorstore   向量库（chromadb，可降级为 numpy）
    retriever     关键词 + 向量混合检索
    generator     提示词构造与 LLM 生成
    pipeline      端到端编排与命令行入口
"""

__version__ = "0.1.0"
