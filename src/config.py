"""集中配置。

所有可调参数都从项目根目录的 ``.env`` 读取（参考 ``.env.example``）。
任何一项缺失都使用下面的默认值，因此**不做任何配置也能用 mock 语料跑通全流程**。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # 这些字段**绝不能出现在日志、异常、测试报告里**。默认的 dataclass/pydantic repr
    # 会把值原样打出来——实测踩过：pytest 失败时会把 fixture 的取值连同报告一起打印，
    # 于是完整的 WEIBO_COOKIE 直接落到了终端（等于账号会话）。
    SECRET_FIELDS: ClassVar[tuple[str, ...]] = ("weibo_cookie", "llm_api_key")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._masked_summary()})"

    def __str__(self) -> str:  # pragma: no cover - 与 repr 同源，保留一致性
        return self.__repr__()

    def _masked_summary(self) -> str:
        parts: list[str] = []
        for name in type(self).model_fields:
            value = getattr(self, name, None)
            if name in self.SECRET_FIELDS:
                length = len(str(value or "").strip())
                parts.append(f"{name}={'<未设置>' if not length else f'<已设置 {length} 字符>'}")
            else:
                parts.append(f"{name}={value!r}")
        return ", ".join(parts)

    # ---------- 微博采集 ----------
    weibo_cookie: str = ""
    crawl_interval: float = 2.5          # 两次请求之间的最小间隔（秒），务必别再调小
    crawl_timeout: float = 15.0
    crawl_max_retry: int = 3
    crawl_pages: int = 3                 # 每个关键词翻页数
    http_proxy: str = ""                 # 例如 http://127.0.0.1:7890

    # ---------- 每日定时更新 ----------
    # 每晚重扫这些关键词的**最前面几页**（新内容在搜索结果顶部），靠博文 id 去重。
    # 注意：这与「补历史」是两种不同的采集语义，不要混用同一个页码游标。
    daily_keywords: str = "Lolita,lo裙"    # 逗号分隔
    daily_pages: int = 2
    daily_lock_stale_hours: float = 6.0    # 锁超过该时长视为陈旧，允许被接管

    # ---------- LLM（OpenAI 兼容协议）----------
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_timeout: float = 60.0
    llm_max_tokens: int = 1024

    # ---------- Embedding ----------
    # auto：优先 BGE，失败自动降级为 hash；bge：强制 BGE；hash：强制本地哈希向量
    embedding_backend: Literal["auto", "bge", "hash"] = "auto"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_allow_download: bool = False   # 默认只用本地缓存，避免离线时长时间超时
    embedding_batch_size: int = 32
    embedding_query_prefix: str = "为这个句子生成表示以用于检索相关文章："

    # ---------- 检索 ----------
    vector_backend: Literal["chroma", "simple"] = "chroma"
    chroma_collection: str = "lolita_posts"
    top_k: int = 5          # 最终送进 prompt 的片段数
    recall_k: int = 20      # 每路召回条数（BM25 / 向量各取这么多）
    rrf_k: int = 60         # RRF 融合常数（仅 fusion_mode=rrf 时使用）
    bm25_weight: float = 1.0
    vector_weight: float = 1.0

    # 融合方式：
    #   weighted —— 每路先按本查询内的最高分归一化再线性加权，**保留「好多少」的信息**；
    #   rrf      —— 只用名次（Reciprocal Rank Fusion），对分数尺度不敏感但会压平差距。
    # 小候选集下 RRF 会把「BM25 给 7.15 分」和「只匹配了一个常见词拿 1.08 分」压成 1/61 与 1/63，
    # 于是噪声被抬进 top-k；这是实测出来的问题，所以默认用 weighted。
    fusion_mode: Literal["weighted", "rrf"] = "weighted"

    # 向量相似度的**绝对**下限。低于它的结果根本不进入融合，也不参与 RRF——
    # 这是让系统能说出「语料里没有」的唯一有效手段（稠密检索永远会返回最近的若干条）。
    #
    # 0.44 是用 scripts/explain_retrieval.py --scan 在 BGE-small-zh 上标定的。
    # ⚠️ 这个值**不是**「相关与不相关之间的空隙」——690 条语料上两者已经重叠
    # （相关最低 0.4602，语料外最高 0.5049），单靠阈值分不开。现在它只是
    # 「进结果的最低语义门槛」，真正的精度由另外两道闸门提供：
    # 内容词要求（对每一路生效）与关键词路的两词证据（见 retriever/hybrid.py）。
    # 实测 0.44 比 0.45 好：0.45 会误拒一条只靠关键词路找到的正确博文
    # （Recall@5 0.909 / MRR 0.745，而 0.44 是 1.000 / 0.836）。
    # ⚠️ 换 embedding 模型或语料规模明显变化后必须重新标定，别沿用这个数。
    min_vector_score: float = 0.44

    # BM25 分数的**相对**下限：低于「本查询最高分 × 该比例」的命中被丢弃。
    # BM25 的绝对值随语料规模变化（IDF 会涨），所以只能用相对值。
    # 作用是剪掉「只匹配了裙子/图片这类常见词」的尾巴。
    min_bm25_ratio: float = 0.2

    # 「生僻词」判定：某个查询词在语料里出现频率 ≤ 该比例时算生僻词（有区分度）。
    #
    # 为什么需要它：内容词闸门只要求「命中至少一个长度≥2 的词」，而「裙子」「lo裙」
    # 「日常」这类词在 681 个片段里出现上百次，命中它们几乎不提供任何信息。实测
    # 「藤上春这条裙子多少钱」的 top-5 里有 4 条只命中了「裙子」，与藤上春毫无关系。
    # 所以规则改成：**查询里只要存在生僻词，片段就必须命中至少一个生僻词**；
    # 全是常见词时才退回到「命中任意内容词」。
    # 0.10 的依据：实测「痛lo」df=36/681（5.3%，是真主题词）而「裙子」远超 10%；
    # 取 10% 能保住前者、剪掉后者。
    distinctive_df_ratio: float = 0.10

    # ---------- 切分 ----------
    # 短于该字数的博文不进入索引：它只有一个短语，构不成可检索的知识单元，
    # 却会在 BM25 里拿到很高的 idf 密度（实测「就这样幸福的痛lo」这类 15 字贴
    # 曾以 1.362 的融合分排到第 1）。实测 681 片段里 <20 字的有 81 条（11.9%），
    # 而评测集里 11 个正确答案全部是 69~150 字——砍掉短贴不会伤到真答案。
    chunk_min_len: int = 18

    # ---------- 分词 ----------
    # jieba 自定义词典：Lolita 圈的黑话（柄图/痛lo/茶会…）不在主词典里，
    # 不加的话会被切成单字，稀释 BM25 的区分度。改完即时生效，无需重建索引。
    jieba_userdict: Path = PROJECT_ROOT / "config" / "jieba_userdict.txt"

    # ---------- 查询日志 ----------
    # 每次**真实**检索落一行 jsonl（`data/logs/queries.jsonl`）。
    #
    # 为什么必须有：评测集是「读完语料手写」的，拿它做优化等于自己出题自己答题——
    # 指标涨了也可能只是拟合了那几十道题。线上真实查询是唯一不受这层影响的证据，
    # 也是「发现 bad case」的唯一闭环（没有它就只能靠人坐下来编查询）。
    #
    # 两条硬约束（实现见 pipeline/query_log.py）：
    #   1. 写日志失败绝不能让检索失败；
    #   2. 评测与排查脚本直接调 retriever，**不进日志**，否则会污染拒答率这类统计。
    query_log_enabled: bool = True
    # 每行只存前 N 条命中的摘要（chunk_id / post_id / 作者 / 分数 / 召回路径），
    # 存全文会让日志迅速膨胀，而排查时看摘要已经够定位。
    query_log_top_hits: int = 3

    # ---------- 目录与日志 ----------
    data_dir: Path = PROJECT_ROOT / "data"
    log_level: str = "INFO"

    @field_validator("data_dir", mode="before")
    @classmethod
    def _empty_data_dir_means_default(cls, value: object) -> object:
        """``DATA_DIR=``（空值）必须回落到项目下的 data/。

        否则 pydantic 会把空字符串解析成 ``Path('.')``，
        数据、索引、日志就全都写进项目根目录了。
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return PROJECT_ROOT / "data"
        return value

    # ---------- 派生路径 ----------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        """索引元信息：记录实际使用的向量后端、embedding 模型与维度。

        没有它就无法判断「当前 .env 的模型」和「已建好的索引」是否匹配。
        """
        return self.index_dir / "manifest.json"

    @property
    def raw_posts_path(self) -> Path:
        return self.raw_dir / "posts.jsonl"

    @property
    def crawl_state_path(self) -> Path:
        """采集进度账本，支持断点续爬。"""
        return self.raw_dir / "crawl_state.json"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def history_path(self) -> Path:
        """每次定时更新的单行汇总，便于看「每天新增多少条」的趋势。"""
        return self.logs_dir / "history.jsonl"

    @property
    def query_log_path(self) -> Path:
        """真实查询日志：每次检索一行。评测/排查脚本直接调 retriever，不写这里。"""
        return self.logs_dir / "queries.jsonl"

    @property
    def lock_path(self) -> Path:
        return self.data_dir / ".daily_update.lock"

    @property
    def daily_keyword_list(self) -> list[str]:
        # 中文用户很容易打成全角逗号，这里一并兼容
        raw = self.daily_keywords.replace("，", ",")
        return [item.strip() for item in raw.split(",") if item.strip()]

    @property
    def simple_vectors_path(self) -> Path:
        return self.index_dir / "vectors.npz"

    @property
    def embedding_cache_path(self) -> Path:
        return self.cache_dir / "embeddings.pkl"

    @property
    def chroma_path(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def llm_enabled(self) -> bool:
        """没有配置 API Key 时走抽取式兜底答案，保证离线可用。"""
        return bool(self.llm_api_key.strip())

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.index_dir,
            self.cache_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
