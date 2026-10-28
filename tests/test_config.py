"""config.py: defaults, env overrides, int parsing fallback."""
import config
from config import Settings

ENV_KEYS = [
    "DATA_DIR", "CHUNKS_DIR", "QA_DIR", "CHROMA_DB_PATH",
    "EMBED_MODEL", "RERANKER_MODEL", "LLM_MODEL", "QA_GEN_MODEL",
    "LLM_BASE_URL", "LLM_API_KEY",
    "COLLECTION_NAME", "TOP_K", "RERANK_TOP_K",
    "CHUNK_TOKENS", "OVERLAP_TOKENS", "CORS_ORIGINS",
]


def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults(monkeypatch):
    clean_env(monkeypatch)
    s = Settings()
    assert s.embed_model == "BAAI/bge-m3"
    assert s.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert s.top_k == 5
    assert s.rerank_top_k == 4
    assert s.chunk_tokens == 512
    assert s.overlap_tokens == 64
    assert s.collection_name == "ming_military"
    assert s.cors_origins == ["http://localhost:3000", "http://localhost:8081"]
    assert str(s.data_dir) == "data"


def test_env_overrides(monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setenv("TOP_K", "8")
    monkeypatch.setenv("CHUNK_TOKENS", "1024")
    monkeypatch.setenv("COLLECTION_NAME", "ming_military_new")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.example , http://b.example")
    s = Settings()
    assert s.top_k == 8
    assert s.chunk_tokens == 1024
    assert s.collection_name == "ming_military_new"
    assert s.cors_origins == ["http://a.example", "http://b.example"]


def test_invalid_int_falls_back_to_default(monkeypatch):
    clean_env(monkeypatch)
    monkeypatch.setenv("TOP_K", "not-a-number")
    s = Settings()
    assert s.top_k == 5


def test_env_variable_lookup_helpers(monkeypatch):
    clean_env(monkeypatch)
    assert config._env("MISSING_KEY", "fallback") == "fallback"
    assert config._env_int("MISSING_INT", 7) == 7
    monkeypatch.setenv("MISSING_KEY", "value")
    assert config._env("MISSING_KEY", "fallback") == "value"


def test_settings_instance_exists():
    assert hasattr(config, "settings")
