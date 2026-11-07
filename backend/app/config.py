"""Central configuration. Every value can be overridden with an environment variable
(or a .env file), e.g. LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.1."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM -------------------------------------------------------------
    llm_provider: Literal["openai", "groq", "ollama", "fake"] = "openai"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"  # LLaMA hosted on Groq (free tier available)
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0

    # --- Embeddings ------------------------------------------------------
    # "hf" runs a BGE / GTE model locally through sentence-transformers.
    embeddings_provider: Literal["hf", "fake"] = "hf"
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # or thenlper/gte-large, BAAI/bge-large-en-v1.5
    embedding_device: str = "cpu"

    # --- Vector store ----------------------------------------------------
    chroma_dir: str = "./data/chroma"
    collection_name: str = "docpilot"

    # --- RAG behaviour ---------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    max_rewrites: int = 2

    # --- API -------------------------------------------------------------
    cors_origins: str = "http://localhost:5173"
    allow_uploads: bool = True
    max_upload_mb: int = 5
    auto_ingest_samples: bool = True  # index data/sample_docs on startup if the store is empty


@lru_cache
def get_settings() -> Settings:
    return Settings()
