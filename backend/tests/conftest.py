"""Tests run fully offline: fake LLM + hashing embeddings + a throwaway ChromaDB folder."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_TMP = tempfile.mkdtemp(prefix="docpilot-test-")
os.environ.update(
    {
        "LLM_PROVIDER": "fake",
        "EMBEDDINGS_PROVIDER": "fake",
        "CHROMA_DIR": _TMP,
        "COLLECTION_NAME": "test",
    }
)

SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "data" / "sample_docs"


@pytest.fixture(scope="session", autouse=True)
def ingested():
    from app.ingest import ingest_dir

    counts = ingest_dir(SAMPLE_DOCS)
    yield counts
    shutil.rmtree(_TMP, ignore_errors=True)
