"""Policy agent: Retrieval-Augmented Generation with citations.

LLM mode: retrieved chunks are numbered and passed as context; the model must
answer only from that context and cite chunk numbers, or say it doesn't know.

Offline mode: extractive answering. Sentences from the retrieved chunks are
scored by term overlap with the question and the best ones are returned
verbatim with their citations, so answers are always grounded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..llm import LLMClient
from ..observability.tracing import Tracer
from ..retrieval.bm25 import tokenize
from ..retrieval.retriever import Hit, Retriever

SYSTEM = (
    "You are a retail operations assistant for store associates. Answer ONLY from the numbered "
    "context. Cite sources like [1] or [2]. If the context does not contain the answer, reply "
    "exactly: I don't know based on current policy documents. Be concise (2-4 sentences)."
)
NO_ANSWER = "I don't know based on current policy documents."


@dataclass
class PolicyAnswer:
    answer: str
    citations: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)


def _extractive(question: str, hits: list[Hit], max_sentences: int = 3) -> tuple[str, list[str]]:
    q_terms = set(tokenize(question))
    candidates = []
    for rank, hit in enumerate(hits):
        for sent in re.split(r"(?<=[.!?])\s+", hit.chunk.text):
            overlap = len(q_terms & set(tokenize(sent)))
            if overlap:
                # prefer higher-ranked chunks on ties
                candidates.append((overlap - 0.5 * rank, sent.strip(), hit.chunk.citation))
    if not candidates:
        return NO_ANSWER, []
    best = sorted(candidates, key=lambda c: -c[0])[:max_sentences]
    citations = list(dict.fromkeys(c[2] for c in best))
    text = " ".join(f"{s} [{citations.index(cit) + 1}]" for _, s, cit in best)
    return text, citations


class PolicyAgent:
    def __init__(self, retriever: Retriever, llm: Optional[LLMClient], tracer: Tracer,
                 mode: str = "hybrid", k: int = 4):
        self.retriever, self.llm, self.tracer, self.mode, self.k = retriever, llm, tracer, mode, k

    def answer(self, question: str) -> PolicyAnswer:
        with self.tracer.span("policy_agent") as span:
            hits = self.retriever.search(question, k=self.k, mode=self.mode, tracer=self.tracer)
            ids = [h.chunk.chunk_id for h in hits]
            contexts = [h.chunk.text for h in hits]
            if self.llm is None:
                text, cites = _extractive(question, hits)
            else:
                context = "\n\n".join(f"[{i + 1}] ({h.chunk.citation}) {h.chunk.text}" for i, h in enumerate(hits))
                with self.tracer.span("llm_generate"):
                    text = self.llm.complete(SYSTEM, f"Context:\n{context}\n\nQuestion: {question}").text.strip()
                used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
                cites = [hits[n - 1].chunk.citation for n in used]
            span.attrs["citations"] = cites
            return PolicyAnswer(text, cites, ids, contexts)
