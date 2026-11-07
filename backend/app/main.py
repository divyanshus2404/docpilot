"""FastAPI service exposing the DocPilot agent.

POST /api/documents        upload .pdf/.md/.txt files (chunked, embedded, stored in ChromaDB)
GET  /api/documents        list ingested documents
DELETE /api/documents/{s}  remove a document
POST /api/chat             ask a question -> answer, citations, agent trace
POST /api/chat/stream      same, streamed as Server-Sent Events (agent steps, then answer tokens)
GET  /api/health           provider info
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager

from .config import get_settings
from .graph import get_graph, turn_input
from .ingest import SUPPORTED, ingest_dir, ingest_file
from .vectorstore import delete_source, list_sources

settings = get_settings()
SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "data" / "sample_docs"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Fresh deployments (e.g. Hugging Face Spaces) start with an empty store: seed the demo docs.
    if settings.auto_ingest_samples and not list_sources() and SAMPLE_DOCS.is_dir():
        ingest_dir(SAMPLE_DOCS)
    yield


app = FastAPI(title="DocPilot", version="1.0.0", description="Agentic RAG over your documents", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None  # reuse to keep conversation memory


class Citation(BaseModel):
    id: int
    source: str
    page: int | None
    score: float
    content: str


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    citations: list[Citation]
    steps: list[str]


@app.get("/api/health")
def health():
    model = {"openai": settings.openai_model, "groq": settings.groq_model, "ollama": settings.ollama_model}.get(
        settings.llm_provider, "fake"
    )
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": model,
        "uploads_enabled": settings.allow_uploads,
        "embedding_model": settings.embedding_model if settings.embeddings_provider == "hf" else "hashing (offline)",
    }


@app.post("/api/documents")
async def upload(files: list[UploadFile] = File(...)):
    if not settings.allow_uploads:
        raise HTTPException(403, "Uploads are disabled on this demo")
    results = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in SUPPORTED:
            raise HTTPException(400, f"{f.filename}: only {', '.join(sorted(SUPPORTED))} are supported")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(f.file, tmp)
            tmp_path = Path(tmp.name)
        if tmp_path.stat().st_size > settings.max_upload_mb * 1024 * 1024:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(413, f"{f.filename}: files must be under {settings.max_upload_mb} MB")
        try:
            n = ingest_file(tmp_path, source_name=f.filename)
        finally:
            tmp_path.unlink(missing_ok=True)
        results.append({"source": f.filename, "chunks": n})
    return {"ingested": results}


@app.get("/api/documents")
def documents():
    return {"documents": list_sources()}


@app.delete("/api/documents/{source}")
def remove(source: str):
    if not settings.allow_uploads:
        raise HTTPException(403, "Editing documents is disabled on this demo")
    removed = delete_source(source)
    if not removed:
        raise HTTPException(404, f"No document named {source}")
    return {"removed_chunks": removed}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    result = await get_graph().ainvoke(turn_input(req.question), {"configurable": {"thread_id": thread_id}})
    return ChatResponse(
        thread_id=thread_id,
        answer=result["answer"],
        citations=[Citation(**c) for c in result.get("citations", [])],
        steps=result.get("steps", []),
    )


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def events():
        yield _sse("thread", {"thread_id": thread_id})
        final: dict = {}
        streamed_tokens = False
        async for mode, payload in get_graph().astream(
            turn_input(req.question), config, stream_mode=["updates", "messages"]
        ):
            if mode == "updates":
                for node, update in payload.items():
                    if update and update.get("steps"):
                        yield _sse("step", {"node": node, "text": update["steps"][-1]})
                    if update:
                        final.update(update)
            elif mode == "messages":
                chunk, meta = payload
                # Only real token chunks from the LLM; skip whole messages written to state.
                if meta.get("langgraph_node") == "generate" and isinstance(chunk, AIMessageChunk) and chunk.content:
                    streamed_tokens = True
                    yield _sse("token", {"text": chunk.content})
        if not streamed_tokens and final.get("answer"):
            yield _sse("token", {"text": final["answer"]})
        yield _sse("done", {"answer": final.get("answer", ""), "citations": final.get("citations", [])})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# Serve the built React app (frontend/dist) from the same server, e.g. inside Docker.
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
