"""Document ingestion: load -> split into overlapping chunks -> embed -> upsert into ChromaDB.

Chunk IDs are content hashes, so re-ingesting the same file never creates duplicates.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import get_settings
from .vectorstore import delete_source, get_vectorstore

SUPPORTED = {".pdf", ".md", ".txt"}


def load_file(path: Path, source_name: str | None = None) -> list[Document]:
    source = source_name or path.name
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [
            Document(page_content=page.extract_text() or "", metadata={"source": source, "page": i + 1})
            for i, page in enumerate(reader.pages)
        ]
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": source, "page": 1})]


def split(docs: list[Document]) -> list[Document]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    for c in chunks:
        c.metadata["chunk_id"] = hashlib.sha1(
            f"{c.metadata['source']}|{c.metadata.get('page')}|{c.page_content}".encode()
        ).hexdigest()
    return chunks


def ingest_file(path: Path, source_name: str | None = None, replace: bool = True) -> int:
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    source = source_name or path.name
    if replace:
        delete_source(source)  # re-uploading a file replaces its old chunks
    chunks = split(load_file(path, source))
    if not chunks:
        return 0
    get_vectorstore().add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return len(chunks)


def ingest_dir(directory: Path) -> dict[str, int]:
    return {
        p.name: ingest_file(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into DocPilot")
    parser.add_argument("path", type=Path, help="File or folder of .pdf/.md/.txt files")
    args = parser.parse_args()
    result = ingest_dir(args.path) if args.path.is_dir() else {args.path.name: ingest_file(args.path)}
    for name, n in result.items():
        print(f"{name}: {n} chunks")
