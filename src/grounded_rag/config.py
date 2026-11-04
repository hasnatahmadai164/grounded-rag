from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process configuration, read from the environment and an optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    log_level: str = "INFO"
    log_json: bool = True
    log_content: bool = Field(
        default=False,
        description="Log raw questions and answers. Off by default because they may be private.",
    )

    database_url: str = "postgresql+asyncpg://grounded_rag:grounded_rag@localhost:5432/grounded_rag"
    database_pool_size: int = Field(default=5, ge=1)

    default_corpus: str = "default"

    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=1)
    embedding_batch_size: int = Field(default=96, ge=1, le=2048)
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    llm_send_temperature: bool = True
    llm_max_output_tokens: int = Field(default=1024, ge=16)

    chat_input_usd_per_mtok: float = Field(default=0.15, ge=0)
    chat_output_usd_per_mtok: float = Field(default=0.60, ge=0)
    embedding_usd_per_mtok: float = Field(default=0.02, ge=0)

    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = "grounded-rag"
    pinecone_index_host: str | None = Field(
        default=None,
        description="Data-plane host of the index. Looked up by name when unset.",
    )
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_api_version: str = "2025-01"
    pinecone_timeout_seconds: float = Field(default=15.0, gt=0)
    pinecone_max_retries: int = Field(default=3, ge=0)

    chunk_target_chars: int = Field(default=1200, ge=100)
    chunk_max_chars: int = Field(default=1800, ge=100)
    chunk_min_chars: int = Field(default=200, ge=0)
    chunk_overlap_chars: int = Field(default=150, ge=0)

    default_top_k: int = Field(default=6, ge=1)
    max_top_k: int = Field(default=20, ge=1)
    retrieval_dense_top_k: int = Field(default=30, ge=1)
    retrieval_lexical_top_k: int = Field(default=30, ge=1)
    retrieval_rrf_k: int = Field(default=60, ge=1)
    retrieval_candidates: int = Field(default=20, ge=1)

    request_token_budget: int = Field(default=60_000, ge=1)
    request_cost_budget_usd: float = Field(default=0.05, gt=0)

    eval_cache_dir: Path = Path(".cache/llm")

    rerank_enabled: bool = True
    agent_max_rewrites: int = Field(default=2, ge=0)
    agent_max_repairs: int = Field(default=1, ge=0)
    agent_max_context_chunks: int = Field(default=10, ge=1)

    request_timeout_seconds: float = Field(default=60.0, gt=0)

    judge_model: str | None = Field(
        default=None,
        description="Model that grades answers in evaluations. Defaults to CHAT_MODEL.",
    )
    eval_corpus: str = "eval-cuad"

    max_request_bytes: int = Field(default=65_536, ge=1024)
