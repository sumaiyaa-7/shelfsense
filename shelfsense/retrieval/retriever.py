"""Retriever with three modes: lexical (BM25), dense (vector), and hybrid.

Hybrid uses Reciprocal Rank Fusion (RRF): each chunk's score is the sum of
1 / (k + rank) across retrievers. RRF needs no score calibration between
BM25 and cosine similarity, which is why it is a common production default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..observability.tracing import Tracer
from .bm25 import BM25
from .chunker import Chunk, load_corpus
from .embeddings import Embedder, LSAEmbedder

MODES = ("bm25", "vector", "hybrid")


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Optional[Embedder] = None, rrf_k: int = 60):
        self.chunks = chunks
        texts = [f"{c.title}. {c.section}. {c.text}" for c in chunks]
        self.bm25 = BM25(texts)
        self.embedder = (embedder or LSAEmbedder()).fit(texts)
        self.matrix = self.embedder.embed(texts)  # in-memory vector index (rows are unit vectors)
        self.rrf_k = rrf_k

    @classmethod
    def from_dir(cls, policies_dir: Path, embedder: Optional[Embedder] = None) -> "Retriever":
        return cls(load_corpus(policies_dir), embedder)

    def _rank_bm25(self, query: str) -> list[int]:
        s = np.array(self.bm25.scores(query))
        return [int(i) for i in np.argsort(-s) if s[i] > 0]

    def _rank_vector(self, query: str) -> list[int]:
        sims = self.matrix @ self.embedder.embed([query])[0]
        return [int(i) for i in np.argsort(-sims)]

    def search(self, query: str, k: int = 4, mode: str = "hybrid", tracer: Optional[Tracer] = None) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        tracer = tracer or Tracer()
        with tracer.span("retrieval", mode=mode, k=k) as span:
            if mode == "bm25":
                ranked = self._rank_bm25(query)
                hits = [Hit(self.chunks[i], 1 / (r + 1)) for r, i in enumerate(ranked[:k])]
            elif mode == "vector":
                ranked = self._rank_vector(query)
                hits = [Hit(self.chunks[i], 1 / (r + 1)) for r, i in enumerate(ranked[:k])]
            else:
                fused: dict[int, float] = {}
                for ranking in (self._rank_bm25(query), self._rank_vector(query)):
                    for rank, idx in enumerate(ranking):
                        fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
                hits = [Hit(self.chunks[i], s) for i, s in top]
            span.attrs["hits"] = [h.chunk.chunk_id for h in hits]
            return hits
