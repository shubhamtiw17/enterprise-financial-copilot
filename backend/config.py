from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "Enterprise Financial Copilot"
    app_env: str = "development"

    # LLM Provider — switch between "openai" and "bedrock"
    llm_provider: str = "openai"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # AWS Bedrock
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    # Database
    database_url: str = "postgresql://copilot:copilot@localhost:5432/copilot"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # RAG settings
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k_results: int = 6

    # LangSmith (observability)
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "false"
    langchain_project: str = "financial-copilot"

    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache
def get_settings() -> Settings:
    return Settings()