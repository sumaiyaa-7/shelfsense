"""Orchestrator: the entry point that ties router, agents, and tracing together."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from ..config import Settings, get_settings
from ..llm import LLMClient, get_llm
from ..observability.tracing import Tracer
from ..retrieval.retriever import Retriever
from .data_agent import DataAgent
from .policy_agent import PolicyAgent
from .router import route


@dataclass
class CopilotResponse:
    question: str
    route: str
    answer: str
    citations: list[str] = field(default_factory=list)
    sql: Optional[str] = None
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    trace_id: str = ""
    latency_ms: float = 0.0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class Copilot:
    def __init__(self, settings: Optional[Settings] = None, llm: Optional[LLMClient] = "auto",
                 retrieval_mode: Optional[str] = None, tracer: Optional[Tracer] = None):
        self.settings = settings or get_settings()
        self.llm = get_llm() if llm == "auto" else llm
        self.tracer = tracer or Tracer(self.settings.trace_path)
        self.retriever = Retriever.from_dir(self.settings.policies_dir)
        mode = retrieval_mode or self.settings.retrieval_mode
        self.policy_agent = PolicyAgent(self.retriever, self.llm, self.tracer, mode=mode, k=self.settings.top_k)
        self.data_agent = DataAgent(self.settings.db_path, self.llm, self.tracer)

    def ask(self, question: str) -> CopilotResponse:
        with self.tracer.span("request", question=question) as root:
            with self.tracer.span("router") as rspan:
                r = route(question, self.llm)
                rspan.attrs.update(route=r.label, reason=r.reason)
            resp = CopilotResponse(question=question, route=r.label, answer="", trace_id=root.trace_id)

            if r.label in ("policy", "hybrid"):
                p = self.policy_agent.answer(question)
                resp.citations, resp.retrieved_ids, resp.contexts = p.citations, p.retrieved_ids, p.contexts
                resp.answer = p.answer
            if r.label in ("data", "hybrid"):
                d = self.data_agent.answer(question)
                resp.sql, resp.columns, resp.rows = d.sql, d.columns, [list(r) for r in d.rows]
                resp.answer = d.answer if r.label == "data" else f"**Policy:** {resp.answer}\n\n**Data:**\n{d.answer}"
            if resp.citations:
                resp.answer += "\n\nSources: " + "; ".join(f"[{i + 1}] {c}" for i, c in enumerate(resp.citations))

        resp.latency_ms = round(root.duration_ms, 2)
        resp.total_tokens = root.total_tokens()
        return resp
