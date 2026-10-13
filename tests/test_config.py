"""配置测试：派生路径与「无 Key 也能用」的判定。"""

from __future__ import annotations

from pathlib import Path

from src.config import PROJECT_ROOT, Settings


def test_derived_paths_live_under_project_root():
    settings = Settings()
    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.processed_dir == settings.data_dir / "processed"
    assert settings.index_dir == settings.data_dir / "index"
    assert settings.manifest_path.name == "manifest.json"
    assert Path(PROJECT_ROOT).is_dir()


def test_llm_enabled_follows_api_key():
    assert Settings(llm_api_key="").llm_enabled is False
    assert Settings(llm_api_key="   ").llm_enabled is False
    assert Settings(llm_api_key="sk-xxx").llm_enabled is True


def test_defaults_are_offline_friendly():
    settings = Settings()
    # 默认不联网下载模型：离线环境下会长时间超时，必须由用户显式打开
    assert settings.embedding_allow_download is False
    # 默认必须「设防」：0.0 等于把稠密检索的噪声全部放进来（实测过的坑）
    assert settings.min_vector_score > 0.0
    assert 0.0 <= settings.min_bm25_ratio < 1.0
    assert settings.fusion_mode in {"weighted", "rrf"}
    # 采集间隔不允许被调成「压测模式」
    assert settings.crawl_interval >= 1.0


def test_empty_data_dir_falls_back_to_project_data():
    """`DATA_DIR=`（空值）必须回落到项目下的 data/，而不是 Path('.')。

    否则数据、索引、日志会全部写进项目根目录——.env.example 里就是这么留空的。
    """
    for blank in ("", "   ", None):
        assert Settings(data_dir=blank).data_dir == Path(PROJECT_ROOT) / "data"


def test_daily_keyword_list_handles_both_comma_styles():
    assert Settings(daily_keywords="Lolita,lo裙").daily_keyword_list == ["Lolita", "lo裙"]
    assert Settings(daily_keywords="Lolita，lo裙 , JSK").daily_keyword_list == ["Lolita", "lo裙", "JSK"]
    assert Settings(daily_keywords="  ,  ").daily_keyword_list == []


def test_logs_and_lock_paths_follow_data_dir(tmp_path):
    settings = Settings(data_dir=tmp_path)
    assert settings.logs_dir == tmp_path / "logs"
    assert settings.history_path == tmp_path / "logs" / "history.jsonl"
    assert settings.lock_path == tmp_path / ".daily_update.lock"
