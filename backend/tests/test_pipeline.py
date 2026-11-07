import uuid

from fastapi.testclient import TestClient

from app.graph import get_graph, route_after_grade, turn_input
from app.main import app
from app.vectorstore import list_sources, search

client = TestClient(app)


def ask(question: str, thread_id: str | None = None) -> dict:
    return get_graph().invoke(turn_input(question), {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}})


# ---------------------------------------------------------------- ingestion & retrieval


def test_ingestion_creates_chunks(ingested):
    assert set(ingested) == {"travel_policy.md", "leave_policy.md", "it_security_policy.md"}
    assert all(n > 0 for n in ingested.values())


def test_reingest_does_not_duplicate(ingested):
    from app.ingest import ingest_file
    from tests.conftest import SAMPLE_DOCS

    before = {d["source"]: d["chunks"] for d in list_sources()}
    ingest_file(SAMPLE_DOCS / "leave_policy.md")
    after = {d["source"]: d["chunks"] for d in list_sources()}
    assert before == after


def test_retrieval_finds_right_document():
    top = search("minimum password length", k=3)
    assert top[0][0].metadata["source"] == "it_security_policy.md"
    assert 0 <= top[0][1] <= 1


# ---------------------------------------------------------------- agent behaviour


def test_answers_with_citation():
    out = ask("How much paid annual leave do full-time employees get?")
    assert "24 days" in out["answer"]
    assert out["citations"] and out["citations"][0]["source"] == "leave_policy.md"
    assert any("Graded" in s for s in out["steps"])


def test_declines_when_not_in_documents():
    out = ask("Who won the cricket world cup in 2011?")
    assert out["citations"] == []
    assert "couldn't find" in out["answer"].lower()
    assert any("retrying search" in s for s in out["steps"]), "should try a query rewrite before giving up"


def test_follow_up_uses_conversation_memory():
    thread = str(uuid.uuid4())
    ask("What is the daily meal allowance for international trips?", thread)
    out = ask("And what about hotels in metro cities?", thread)
    assert any("Follow-up rewritten" in s for s in out["steps"])
    state = get_graph().get_state({"configurable": {"thread_id": thread}}).values
    assert len(state["messages"]) == 4  # 2 user + 2 assistant turns remembered


def test_routing_logic():
    assert route_after_grade({"relevant": [{"content": "x"}]}) == "generate"
    assert route_after_grade({"relevant": [], "rewrites": 0}) == "rewrite"
    assert route_after_grade({"relevant": [], "rewrites": 2}) == "not_found"


# ---------------------------------------------------------------- API


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["llm_provider"] == "fake"


def test_chat_endpoint_and_thread_reuse():
    r = client.post("/api/chat", json={"question": "How quickly must a lost laptop be reported?"})
    body = r.json()
    assert r.status_code == 200
    assert "1 hour" in body["answer"]
    assert body["citations"][0]["source"] == "it_security_policy.md"
    r2 = client.post("/api/chat", json={"question": "And what about phishing emails?", "thread_id": body["thread_id"]})
    assert r2.json()["thread_id"] == body["thread_id"]


def test_stream_endpoint_emits_steps_tokens_and_done():
    with client.stream("POST", "/api/chat/stream", json={"question": "What is the minimum password length?"}) as r:
        text = "".join(r.iter_text())
    assert "event: step" in text and "event: token" in text and "event: done" in text
    assert "14" in text
    assert text.count("event: token") == 1  # answer must not be duplicated


def test_upload_list_delete():
    files = {"files": ("notes.txt", b"The office cafeteria opens at 8 am every weekday.", "text/plain")}
    assert client.post("/api/documents", files=files).json()["ingested"][0]["chunks"] == 1
    assert "notes.txt" in [d["source"] for d in client.get("/api/documents").json()["documents"]]
    assert client.delete("/api/documents/notes.txt").status_code == 200
    assert client.delete("/api/documents/notes.txt").status_code == 404


def test_rejects_unsupported_file():
    files = {"files": ("bad.exe", b"x", "application/octet-stream")}
    assert client.post("/api/documents", files=files).status_code == 400


def test_upload_size_limit(monkeypatch):
    from app import main

    monkeypatch.setattr(main.settings, "max_upload_mb", 0)
    files = {"files": ("big.txt", b"x" * 2048, "text/plain")}
    assert client.post("/api/documents", files=files).status_code == 413


def test_uploads_can_be_disabled(monkeypatch):
    from app import main

    monkeypatch.setattr(main.settings, "allow_uploads", False)
    files = {"files": ("a.txt", b"hello", "text/plain")}
    assert client.post("/api/documents", files=files).status_code == 403
    assert client.delete("/api/documents/leave_policy.md").status_code == 403
