"""Offline stand-ins for the embedding model and the LLM.

They let the whole pipeline (ingest -> retrieve -> grade -> rewrite -> answer) run in CI
and in tests with no API key, no GPU and no model download. They are deliberately simple
but behave sensibly: lexical-overlap embeddings and a rule-based "LLM" that follows the
same prompts the real models receive.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the of to in on for and or is are was were be been it this that with as by at from "
    "what which who how when where do does can i we you my our your me us any much many".split()
)


def _stem(word: str) -> str:
    for suffix in ("ing", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


class HashingEmbeddings(Embeddings):
    """Bag-of-words feature hashing, L2-normalised. Similar wording -> similar vectors."""

    def __init__(self, size: int = 512):
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        for tok in tokens(text):
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.size
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _section(text: str, name: str) -> str:
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else ""


class FakeRAGChatModel(BaseChatModel):
    """Rule-based chat model that understands DocPilot's four prompt types."""

    @property
    def _llm_type(self) -> str:
        return "fake-rag"

    def _generate(self, messages: list[BaseMessage], stop: Any = None, **kwargs: Any) -> ChatResult:
        system = " ".join(m.content for m in messages if m.type == "system")
        user = messages[-1].content if messages else ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._respond(system, user)))])

    def _respond(self, system: str, user: str) -> str:
        if "TASK: CONDENSE" in system:
            return _section(user, "follow_up") or user
        if "TASK: GRADE" in system:
            q = set(tokens(_section(user, "question")))
            d = set(tokens(_section(user, "document")))
            return "yes" if q and len(q & d) / len(q) >= 0.3 else "no"
        if "TASK: REWRITE" in system:
            q = _section(user, "question")
            return " ".join(tokens(q)) or q
        if "TASK: ANSWER" in system:
            q = set(tokens(_section(user, "question")))
            context = _section(user, "context")
            best, best_score = None, -1
            for n, block in re.findall(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\Z)", context, re.S):
                sentences = re.split(r"(?<=[.!?])\s+", block.strip())
                for sent in sentences:
                    score = len(q & set(tokens(sent)))
                    if score > best_score:
                        best, best_score = (sent.strip(), n), score
            if not best or best_score == 0:
                return "I couldn't find that in the provided documents."
            return f"{best[0]} [{best[1]}]"
        return "OK"
