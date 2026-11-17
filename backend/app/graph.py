"""The DocPilot agent, built as a LangGraph state machine (corrective / self-checking RAG).

    START -> condense -> retrieve -> graph_expand -> grade --(relevant chunks found)--> generate -> END
                            ^          |
                            |          +--(nothing relevant, retries left)--> rewrite --+
                            |                                                           |
                            +-----------------------------------------------------------+
                                       |
                                       +--(nothing relevant, no retries left)--> not_found -> END

* condense  - turns a follow-up ("what about interns?") into a standalone question using chat memory
* retrieve  - vector search in ChromaDB (BGE embeddings)
* graph_expand - follows the knowledge graph from entities named in the question (GraphRAG)
* grade     - the LLM checks each retrieved chunk for relevance and drops the noise
* rewrite   - if nothing survived grading, the LLM rewrites the search query and we try again
* generate  - answers only from the relevant chunks, with [n] citations
* not_found - refuses instead of hallucinating

Conversation memory is kept per `thread_id` by a LangGraph checkpointer.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import prompts
from .config import get_settings
from .providers import get_llm
from .vectorstore import search


class Chunk(TypedDict, total=False):
    content: str
    source: str
    page: int | None
    score: float
    via: str  # set when the chunk was reached through the knowledge graph


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str  # standalone version of the latest user question
    query: str  # what we actually send to the vector store (may be rewritten)
    documents: list[Chunk]  # raw retrieval results
    relevant: list[Chunk]  # chunks that passed grading
    rewrites: int
    answer: str
    citations: list[Chunk]
    steps: list[str]  # human-readable trace, handy for the UI and for debugging


def _text(msg: AnyMessage) -> str:
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def _step(state: AgentState, text: str) -> list[str]:
    return [*state.get("steps", []), text]


# ----------------------------------------------------------------------------- nodes


def condense(state: AgentState) -> AgentState:
    msgs = state["messages"]
    latest = _text(msgs[-1])
    history = [m for m in msgs[:-1] if m.type in ("human", "ai")][-6:]
    if not history:
        return {"question": latest, "query": latest, "steps": _step(state, "Question received")}

    transcript = "\n".join(f"{'User' if m.type == 'human' else 'Assistant'}: {_text(m)}" for m in history)
    standalone = get_llm().invoke(
        [
            SystemMessage(prompts.CONDENSE_SYSTEM),
            HumanMessage(prompts.CONDENSE_USER.format(history=transcript, question=latest)),
        ]
    ).content.strip()
    return {
        "question": standalone,
        "query": standalone,
        "steps": _step(state, f"Follow-up rewritten as: “{standalone}”"),
    }


def retrieve(state: AgentState) -> AgentState:
    results = search(state["query"])
    docs: list[Chunk] = [
        {
            "content": d.page_content,
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page"),
            "score": score,
        }
        for d, score in results
    ]
    return {"documents": docs, "steps": _step(state, f"Retrieved {len(docs)} chunks for “{state['query']}”")}


def graph_expand(state: AgentState) -> AgentState:
    """Add chunks reached through the knowledge graph from entities named in the question."""
    from .kg import graph_chunks

    found, seeds = graph_chunks(state["question"])
    seen = {d["content"] for d in state.get("documents", [])}
    extra = [c for c in found if c["content"] not in seen]
    if not seeds:
        return {"steps": _step(state, "Graph: no known entities in the question")}
    return {
        "documents": [*state.get("documents", []), *extra],
        "steps": _step(state, f"Graph: matched {', '.join(seeds[:3])}; added {len(extra)} linked chunks"),
    }


def _with_path(d: Chunk) -> str:
    return f"(reached via: {d['via']})\n{d['content']}" if d.get("via") else d["content"]


def grade(state: AgentState) -> AgentState:
    docs = state.get("documents", [])
    if not docs:
        return {"relevant": [], "steps": _step(state, "No chunks to grade")}

    batches = [
        [
            SystemMessage(prompts.GRADE_SYSTEM),
            HumanMessage(prompts.GRADE_USER.format(question=state["question"], document=_with_path(d))),
        ]
        for d in docs
    ]
    verdicts = get_llm().batch(batches, config={"max_concurrency": 5})
    relevant = [d for d, v in zip(docs, verdicts) if _text(v).strip().lower().startswith("yes")]
    return {"relevant": relevant, "steps": _step(state, f"Graded chunks: {len(relevant)}/{len(docs)} relevant")}


def rewrite(state: AgentState) -> AgentState:
    new_query = get_llm().invoke(
        [SystemMessage(prompts.REWRITE_SYSTEM), HumanMessage(prompts.REWRITE_USER.format(question=state["question"]))]
    ).content.strip()
    return {
        "query": new_query,
        "rewrites": state.get("rewrites", 0) + 1,
        "steps": _step(state, f"Nothing relevant, retrying search as: “{new_query}”"),
    }


def generate(state: AgentState) -> AgentState:
    chunks = state["relevant"]
    context = "\n\n".join(
        f"[{i}] (source: {c['source']}, page {c['page']})\n{_with_path(c)}" for i, c in enumerate(chunks, 1)
    )
    answer = get_llm().invoke(
        [
            SystemMessage(prompts.ANSWER_SYSTEM),
            HumanMessage(prompts.ANSWER_USER.format(context=context, question=state["question"])),
        ]
    ).content.strip()

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(chunks)})
    citations = [{**chunks[n - 1], "id": n} for n in cited] or [{**c, "id": i} for i, c in enumerate(chunks, 1)]
    return {
        "answer": answer,
        "citations": citations,
        "messages": [AIMessage(answer)],
        "steps": _step(state, f"Answered from {len(chunks)} chunks"),
    }


def not_found(state: AgentState) -> AgentState:
    return {
        "answer": prompts.NOT_FOUND,
        "citations": [],
        "messages": [AIMessage(prompts.NOT_FOUND)],
        "steps": _step(state, "No relevant information found, declined to guess"),
    }


# ----------------------------------------------------------------------------- routing


def route_after_grade(state: AgentState) -> Literal["generate", "rewrite", "not_found"]:
    if state.get("relevant"):
        return "generate"
    if state.get("rewrites", 0) < get_settings().max_rewrites:
        return "rewrite"
    return "not_found"


# ----------------------------------------------------------------------------- graph


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("condense", condense)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_node("not_found", not_found)

    g.add_edge(START, "condense")
    g.add_edge("condense", "retrieve")
    if get_settings().graph_enabled:
        g.add_node("graph_expand", graph_expand)
        g.add_edge("retrieve", "graph_expand")
        g.add_edge("graph_expand", "grade")
    else:
        g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade)
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("not_found", END)
    return g.compile(checkpointer=checkpointer)


@lru_cache
def get_graph():
    return build_graph(checkpointer=MemorySaver())


def turn_input(question: str) -> AgentState:
    """Fresh per-turn state; `messages` is appended to the thread's stored history."""
    return {"messages": [HumanMessage(question)], "rewrites": 0, "steps": [], "documents": [], "relevant": []}
