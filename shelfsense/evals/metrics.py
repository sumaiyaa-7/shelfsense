"""Per-question metrics for retrieval, routing, generation, and SQL."""
from __future__ import annotations

import sqlite3


def reciprocal_rank(retrieved_ids: list[str], relevant_prefix: str) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid.startswith(relevant_prefix):
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved_ids: list[str], relevant_prefix: str, k: int) -> float:
    return float(any(cid.startswith(relevant_prefix) for cid in retrieved_ids[:k]))


def keyword_recall(answer: str, keywords: list[str]) -> float:
    """Share of required facts (e.g. '14 days') present in the answer."""
    if not keywords:
        return 1.0
    a = answer.lower()
    return sum(k.lower() in a for k in keywords) / len(keywords)


def groundedness(answer: str, contexts: list[str]) -> float:
    """Share of answer sentences whose content words all appear in retrieved context.

    A cheap, deterministic proxy for faithfulness / hallucination rate. In LLM
    mode you would typically add an LLM-as-judge on top of this.
    """
    import re
    from ..retrieval.bm25 import tokenize
    ctx = set(tokenize(" ".join(contexts)))
    body = answer.split("\n\nSources:")[0]
    sents = [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\[\d+\]", "", body)) if tokenize(s)]
    if not sents:
        return 0.0
    return sum(all(t in ctx for t in tokenize(s)) for s in sents) / len(sents)


def _norm(rows, unordered: bool):
    out = [tuple(round(v, 2) if isinstance(v, float) else v for v in r) for r in rows]
    return sorted(out) if unordered else out


def sql_result_match(con: sqlite3.Connection, reference_sql: str, rows: list[tuple], unordered: bool = False) -> float:
    """Execution accuracy: does the agent's result set equal the reference query's?"""
    expected = con.execute(reference_sql).fetchall()
    return float(_norm(expected, unordered) == _norm(rows, unordered))
