"""Model factories. Swapping between GPT (OpenAI) and LLaMA (Groq-hosted or local Ollama) is a single env var,
so the rest of the code never knows which model it is talking to."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import get_settings

# BGE models are trained with an instruction prefix on the *query* side only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache
def get_llm() -> BaseChatModel:
    s = get_settings()
    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=s.openai_model, temperature=s.temperature, api_key=s.openai_api_key)
    if s.llm_provider == "groq":
        # Groq exposes an OpenAI-compatible API, so the same client works.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=s.groq_model,
            temperature=s.temperature,
            api_key=s.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=s.ollama_model, base_url=s.ollama_base_url, temperature=s.temperature)

    from .fakes import FakeRAGChatModel

    return FakeRAGChatModel()


@lru_cache
def get_embeddings() -> Embeddings:
    s = get_settings()
    if s.embeddings_provider == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings

        is_bge = "bge" in s.embedding_model.lower()
        return HuggingFaceEmbeddings(
            model_name=s.embedding_model,
            model_kwargs={"device": s.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={
                "normalize_embeddings": True,
                **({"prompt": BGE_QUERY_INSTRUCTION} if is_bge else {}),
            },
        )

    from .fakes import HashingEmbeddings

    return HashingEmbeddings()
