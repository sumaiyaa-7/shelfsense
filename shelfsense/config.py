"""Central configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    llm_provider: str  # "mock" (offline, deterministic) | "openai" | "anthropic"
    openai_model: str
    anthropic_model: str
    retrieval_mode: str  # "bm25" | "vector" | "hybrid"
    top_k: int
    db_path: Path
    policies_dir: Path
    trace_path: Path


def get_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "mock").lower(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid"),
        top_k=int(os.getenv("TOP_K", "4")),
        db_path=Path(os.getenv("DB_PATH", ROOT / "data" / "retail.db")),
        policies_dir=Path(os.getenv("POLICIES_DIR", ROOT / "data" / "policies")),
        trace_path=Path(os.getenv("TRACE_PATH", ROOT / "traces" / "traces.jsonl")),
    )
