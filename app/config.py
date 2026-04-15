from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_CONFIG_DIR = DATA_DIR / "index_config"
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
VECTOR_STORE_FILE = VECTOR_STORE_DIR / "chunk_index.pkl"

load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Resume LLM RAG Chatbot"
    api_prefix: str = "/api/v1"
    environment: str = Field(default=os.getenv("ENVIRONMENT", "development"))
    huggingfacehub_api_token: str | None = Field(default=None, alias="HUGGINGFACEHUB_API_TOKEN")
    huggingface_repo_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_new_tokens: int = 450
    temperature: float = 0.15
    retriever_top_k: int = 6
    reranker_top_k: int = 6
    semantic_cache_size: int = 64


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
