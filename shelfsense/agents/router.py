"""Router agent: decides which specialist agent(s) handle a question.

- policy -> PolicyAgent (RAG over SOP / policy documents)
- data   -> DataAgent (text-to-SQL over the sales & inventory database)
- hybrid -> both, then the orchestrator merges the answers

With an LLM configured, the LLM classifies the intent; if its output can't be
parsed we fall back to the rule-based classifier rather than failing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..llm import LLMClient
from ..retrieval.bm25 import tokenize

ROUTES = ("policy", "data", "hybrid")

DATA_TERMS = {
    "revenue", "sale", "sold", "top", "best", "selling", "unit", "how many", "total", "trend",
    "month", "by store", "by category", "below reorder", "low stock", "running low", "on hand",
    "compare", "which store", "which product", "which category", "inventory level", "stock level", "q1",
}
POLICY_TERMS = {
    "policy", "return", "refund", "receipt", "price match", "price adjustment", "reward", "allowed",
    "can a customer", "can i", "procedure", "variance", "counterfeit", "safe", "substitut", "pickup",
    "restocking", "expire", "approval", "approve", "cycle count", "shrink", "should", "rule",
    "what happens", "how long", "required", "requires", "drawer",
}

SYSTEM = (
    "You route questions for a retail operations assistant. Reply with exactly one word.\n"
    "policy = answered by store policies / SOPs (returns, price match, rewards, cash handling, "
    "online orders, reorder rules).\n"
    "data = answered by querying sales or inventory numbers.\n"
    "hybrid = needs both numbers and a policy rule."
)


@dataclass
class Route:
    label: str
    reason: str


def _score(question: str, terms: set[str]) -> int:
    q = " " + " ".join(tokenize(question)) + " "
    raw = question.lower()
    return sum(1 for t in terms if t in raw or f" {t} " in q)


def rule_route(question: str) -> Route:
    d, p = _score(question, DATA_TERMS), _score(question, POLICY_TERMS)
    if d and p and min(d, p) >= 1 and re.search(r"\b(and|also|then)\b", question.lower()):
        return Route("hybrid", f"data_score={d}, policy_score={p}, compound question")
    if d > p:
        return Route("data", f"data_score={d} > policy_score={p}")
    return Route("policy", f"policy_score={p} >= data_score={d}")


def route(question: str, llm: Optional[LLMClient] = None) -> Route:
    if llm is None:
        return rule_route(question)
    label = llm.complete(SYSTEM, question, max_tokens=5).text.strip().lower().strip(".")
    if label in ROUTES:
        return Route(label, "llm")
    fallback = rule_route(question)
    return Route(fallback.label, f"llm returned {label!r}; fell back to rules ({fallback.reason})")
