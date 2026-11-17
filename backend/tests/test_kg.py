"""Knowledge-graph layer (GraphRAG): extraction, traversal, multi-hop retrieval, clean-up."""

import uuid

from app.graph import get_graph, turn_input
from app.kg import JsonGraphStore, get_kg, graph_chunks, parse_triples
from app.vectorstore import search

MULTI_HOP = "When was the supplier of the thrust bearing in gas turbine GT-7 last audited?"


def test_parse_triples_ignores_noise():
    text = "GT-7 | uses | B-220\nnot a triple\n- B-220 | supplied by | Kessler Bearings GmbH\n| | x"
    assert parse_triples(text) == [("GT-7", "uses", "B-220"), ("B-220", "supplied by", "Kessler Bearings GmbH")]


def test_store_expand_and_remove(tmp_path):
    kg = JsonGraphStore(tmp_path / "g.json")
    kg.add_chunk("c1", "asset.md", 1, "GT-7 uses B-220", [("GT-7", "uses", "B-220")])
    kg.add_chunk("c2", "catalog.md", 1, "B-220 by Kessler", [("B-220", "supplied by", "Kessler")])
    paths = kg.expand(kg.match_entities("tell me about gt-7"), hops=2)
    assert paths["kessler"] == ["GT-7", "B-220", "Kessler"]
    assert "kessler" not in kg.expand(["gt-7"], hops=1)

    reloaded = JsonGraphStore(tmp_path / "g.json")  # persisted to disk
    assert reloaded.stats() == kg.stats() == {"entities": 3, "relations": 2, "chunks": 2}

    assert kg.remove_source("catalog.md") == 1
    assert "kessler" not in kg.names and kg.stats()["relations"] == 1


def test_entity_match_is_whole_word(tmp_path):
    kg = JsonGraphStore(tmp_path / "g.json")
    kg.add_chunk("c1", "a.md", 1, "", [("GT-7", "uses", "B-220")])
    assert kg.match_entities("status of GT-70?") == []
    assert kg.match_entities("status of GT-7?") == ["gt-7"]


def test_graph_reaches_evidence_that_vector_search_misses():
    vector_sources = {d.metadata["source"] for d, _ in search(MULTI_HOP)}
    chunks, seeds = graph_chunks(MULTI_HOP)
    assert "kessler_bearings.md" not in vector_sources
    assert seeds == ["GT-7"]
    hit = next(c for c in chunks if c["source"] == "kessler_bearings.md")
    assert hit["via"] == "GT-7 → B-220 → Kessler Bearings GmbH"


def test_agent_runs_graph_step():
    out = get_graph().invoke(turn_input(MULTI_HOP), {"configurable": {"thread_id": str(uuid.uuid4())}})
    assert any(s.startswith("Graph: matched GT-7") for s in out["steps"])


def test_no_entities_means_no_graph_chunks():
    assert graph_chunks("How much paid annual leave do employees get?") == ([], [])


def test_deleting_a_document_removes_it_from_the_graph():
    from app.ingest import ingest_file
    from app.vectorstore import delete_source
    from tests.conftest import SAMPLE_DOCS

    path = SAMPLE_DOCS / "plant" / "tellus_valves.md"
    assert any(c["source"] == "tellus_valves.md" for c in get_kg().chunks.values())
    delete_source("tellus_valves.md")
    assert not any(c["source"] == "tellus_valves.md" for c in get_kg().chunks.values())
    ingest_file(path)  # restore for other tests
