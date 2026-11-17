"""ChromaDB access. Uses cosine distance, which is what normalised BGE/GTE embeddings expect."""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import get_settings
from .providers import get_embeddings


@lru_cache
def get_vectorstore() -> Chroma:
    s = get_settings()
    return Chroma(
        collection_name=s.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=s.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def search(query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """Return (document, similarity) pairs, similarity in [0, 1], best first."""
    k = k or get_settings().top_k
    pairs = get_vectorstore().similarity_search_with_score(query, k=k)
    # Chroma returns cosine *distance*; convert to similarity for readability.
    return [(doc, round(1.0 - float(dist), 4)) for doc, dist in pairs]


def list_sources() -> list[dict]:
    data = get_vectorstore().get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"]:
        src = (meta or {}).get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return [{"source": s, "chunks": c} for s, c in sorted(counts.items())]


def delete_source(source: str) -> int:
    from .kg import get_kg  # local import: kg is optional and must not load at import time

    get_kg().remove_source(source)
    vs = get_vectorstore()
    ids = vs.get(where={"source": source})["ids"]
    if ids:
        vs.delete(ids=ids)
    return len(ids)
