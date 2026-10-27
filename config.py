"""
全局配置：所有路径、模型、参数从环境变量读取（支持 .env），脚本参数可覆盖。
用法：from config import settings
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


ROOT_DIR = Path(__file__).resolve().parent


@dataclass
class Settings:
    # 路径
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))
    chunks_dir: Path = field(default_factory=lambda: Path(_env("CHUNKS_DIR", "data/chunks")))
    qa_dir: Path = field(default_factory=lambda: Path(_env("QA_DIR", "data/qa_pairs")))
    chroma_db_path: Path = field(default_factory=lambda: Path(_env("CHROMA_DB_PATH", "data/chroma_db")))

    # 模型
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-m3"))
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3.5:9b"))
    qa_gen_model: str = field(default_factory=lambda: _env("QA_GEN_MODEL", "qwen3.6:35b"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "ollama"))

    # RAG 参数
    collection_name: str = field(default_factory=lambda: _env("COLLECTION_NAME", "ming_military"))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    rerank_top_k: int = field(default_factory=lambda: _env_int("RERANK_TOP_K", 4))

    # 分块参数（token 感知，bge-m3 tokenizer）
    chunk_tokens: int = field(default_factory=lambda: _env_int("CHUNK_TOKENS", 512))
    overlap_tokens: int = field(default_factory=lambda: _env_int("OVERLAP_TOKENS", 64))

    # CORS（逗号分隔的来源列表，默认仅本地）
    cors_origins: List[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in _env("CORS_ORIGINS", "http://localhost:3000,http://localhost:8081").split(",")
            if o.strip()
        ]
    )


settings = Settings()


if __name__ == "__main__":
    for key, value in settings.__dict__.items():
        print(f"{key}={value}")
