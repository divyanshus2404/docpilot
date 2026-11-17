"""Knowledge-graph layer (GraphRAG).

At ingestion, the LLM extracts (subject | relation | object) triples from every chunk. Entities
become nodes, triples become edges, and every entity remembers which chunks mention it.

At question time, `graph_chunks()` finds the entities named in the question, walks up to
`graph_hops` hops through the graph, and returns the chunks that mention the entities it reached,
each tagged with the path that led there (e.g. "GT-7 → B-220 → Kessler Bearings GmbH").

This answers multi-hop questions that plain vector search misses, such as "When was the supplier of
the bearing in turbine GT-7 last audited?". The supplier's audit record never mentions GT-7, so no
embedding of the question lands near it; the graph reaches it through the bearing.

Storage is an in-process graph persisted as JSON behind the small `GraphStore` interface, which is
enough for a single-node deployment. A Neo4j adapter can implement the same interface when the
graph needs to be shared or queried at scale (see docs/ARCHITECTURE.md, ADR-004).
"""

from __future__ import annotations

import json
import re
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from . import prompts
from .config import get_settings
from .providers import get_llm

Triple = tuple[str, str, str]

_MIN_ENTITY_LEN = 3
_MAX_TRIPLES_PER_CHUNK = 20


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}


def norm(name: str) -> str:
    """Canonical entity key: lower-case, single spaces, no surrounding punctuation."""
    return re.sub(r"\s+", " ", name).strip(" .,:;\"'()[]").lower()


class GraphStore(Protocol):
    def add_chunk(self, chunk_id: str, source: str, page: int | None, content: str, triples: list[Triple]) -> None: ...
    def remove_source(self, source: str) -> int: ...
    def match_entities(self, text: str) -> list[str]: ...
    def expand(self, seeds: list[str], hops: int) -> dict[str, list[str]]: ...
    def chunks_mentioning(self, entity: str) -> list[dict]: ...
    def stats(self) -> dict: ...


class JsonGraphStore:
    """Adjacency-list graph kept in memory and saved to a JSON file after every change."""

    def __init__(self, path: Path):
        self.path = path
        self.names: dict[str, str] = {}  # key -> display name
        self.edges: dict[str, dict[str, list[dict]]] = {}  # key -> neighbour key -> [{relation, chunk_id}]
        self.mentions: dict[str, set[str]] = {}  # key -> chunk ids
        self.chunks: dict[str, dict] = {}  # chunk id -> {source, page, content}
        if path.exists():
            self._load()

    # ------------------------------------------------------------------ persistence

    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        self.names = data["names"]
        self.edges = data["edges"]
        self.mentions = {k: set(v) for k, v in data["mentions"].items()}
        self.chunks = data["chunks"]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "names": self.names,
            "edges": self.edges,
            "mentions": {k: sorted(v) for k, v in self.mentions.items()},
            "chunks": self.chunks,
        }
        self.path.write_text(json.dumps(data))

    # ------------------------------------------------------------------ writes

    def _node(self, name: str) -> str | None:
        key = norm(name)
        if len(key) < _MIN_ENTITY_LEN:
            return None
        self.names.setdefault(key, name.strip())
        self.edges.setdefault(key, {})
        self.mentions.setdefault(key, set())
        return key

    def add_chunk(self, chunk_id: str, source: str, page: int | None, content: str, triples: list[Triple]) -> None:
        self.chunks[chunk_id] = {"source": source, "page": page, "content": content}
        for subj, rel, obj in triples[:_MAX_TRIPLES_PER_CHUNK]:
            a, b = self._node(subj), self._node(obj)
            if not a or not b or a == b:
                continue
            self.mentions[a].add(chunk_id)
            self.mentions[b].add(chunk_id)
            edge = {"relation": rel.strip(), "chunk_id": chunk_id}
            self.edges[a].setdefault(b, []).append(edge)
            self.edges[b].setdefault(a, []).append(edge)
        self._save()

    def remove_source(self, source: str) -> int:
        gone = {cid for cid, c in self.chunks.items() if c["source"] == source}
        if not gone:
            return 0
        for cid in gone:
            del self.chunks[cid]
        for key in list(self.mentions):
            self.mentions[key] -= gone
        for key, nbrs in self.edges.items():
            for nbr in list(nbrs):
                nbrs[nbr] = [e for e in nbrs[nbr] if e["chunk_id"] not in gone]
                if not nbrs[nbr]:
                    del nbrs[nbr]
        for key in [k for k in self.names if not self.mentions.get(k) and not self.edges.get(k)]:
            self.names.pop(key, None)
            self.edges.pop(key, None)
            self.mentions.pop(key, None)
        self._save()
        return len(gone)

    # ------------------------------------------------------------------ reads

    def match_entities(self, text: str) -> list[str]:
        """Entities whose name appears in the text as a whole word/phrase, longest first."""
        t = f" {norm(text)} "
        hits = [k for k in self.names if re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", t)]
        return sorted(hits, key=len, reverse=True)

    def expand(self, seeds: list[str], hops: int) -> dict[str, list[str]]:
        """Breadth-first walk. Returns {reached entity: path of display names from a seed}."""
        paths: dict[str, list[str]] = {s: [self.names[s]] for s in seeds if s in self.names}
        queue = deque((s, 0) for s in paths)
        while queue:
            node, depth = queue.popleft()
            if depth >= hops:
                continue
            for nbr in self.edges.get(node, {}):
                if nbr not in paths:
                    paths[nbr] = [*paths[node], self.names[nbr]]
                    queue.append((nbr, depth + 1))
        return paths

    def chunks_mentioning(self, entity: str) -> list[dict]:
        return [{"chunk_id": cid, **self.chunks[cid]} for cid in sorted(self.mentions.get(entity, ())) if cid in self.chunks]

    def stats(self) -> dict:
        n_edges = sum(len(v) for v in self.edges.values()) // 2
        return {"entities": len(self.names), "relations": n_edges, "chunks": len(self.chunks)}


@lru_cache
def get_kg() -> GraphStore:
    s = get_settings()
    return JsonGraphStore(Path(s.graph_path or Path(s.chroma_dir) / "graph.json"))


# ---------------------------------------------------------------------- extraction


def parse_triples(text: str) -> list[Triple]:
    triples = []
    for line in text.splitlines():
        parts = [p.strip(" -*•\t") for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def extract_triples(chunks: list[Document]) -> list[list[Triple]]:
    """One LLM call per chunk, run as a concurrent batch."""
    if not chunks:
        return []
    batches = [
        [SystemMessage(prompts.EXTRACT_SYSTEM), HumanMessage(prompts.EXTRACT_USER.format(chunk=c.page_content))]
        for c in chunks
    ]
    replies = get_llm().batch(batches, config={"max_concurrency": 5})
    return [parse_triples(r.content if isinstance(r.content, str) else str(r.content)) for r in replies]


def index_chunks(chunks: list[Document]) -> int:
    """Extract triples for each chunk and add them to the graph. Returns the number of triples."""
    kg = get_kg()
    total = 0
    for chunk, triples in zip(chunks, extract_triples(chunks)):
        kg.add_chunk(chunk.metadata["chunk_id"], chunk.metadata["source"], chunk.metadata.get("page"), chunk.page_content, triples)
        total += len(triples)
    return total


# ---------------------------------------------------------------------- retrieval


def graph_chunks(question: str, hops: int | None = None, limit: int | None = None) -> tuple[list[dict], list[str]]:
    """Chunks reachable from the entities named in the question.

    Returns (chunks, seed display names). Each chunk carries `via`, the entity path that reached it,
    and a `score` of 1 / path length. Chunks at the far end of a path are ranked first: chunks that
    mention the question's own entities are already found by vector search, so the graph's value
    is the evidence it reaches through other entities.
    """
    s = get_settings()
    hops = s.graph_hops if hops is None else hops
    limit = s.graph_max_chunks if limit is None else limit
    kg = get_kg()
    seeds = kg.match_entities(question)
    if not seeds:
        return [], []

    paths = kg.expand(seeds, hops)
    best: dict[str, dict] = {}
    for entity, path in paths.items():
        for c in kg.chunks_mentioning(entity):
            prev = best.get(c["chunk_id"])
            if prev is None or len(path) > len(prev["_path"]):
                # Prefer the longest path: the far end of the chain is the new evidence.
                best[c["chunk_id"]] = {**c, "_path": path}

    q_words = _words(question)
    ranked = sorted(
        best.values(),
        # Far end of the path first, then the chunks that share the most words with the question.
        key=lambda c: (-len(c["_path"]), -len(q_words & _words(c["content"])), c["source"], c["chunk_id"]),
    )[:limit]
    out = [
        {
            "content": c["content"],
            "source": c["source"],
            "page": c["page"],
            "score": round(1 / len(c["_path"]), 4),
            "via": " → ".join(c["_path"]),
        }
        for c in ranked
    ]
    return out, [kg.names[k] for k in seeds]  # type: ignore[attr-defined]
