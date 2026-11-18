"""Dense embeddings.

- LSAEmbedder: offline default. TF-IDF + truncated SVD (latent semantic
  analysis) gives low-dimensional dense vectors that capture term
  co-occurrence, so it can match "refund" to "return" without an API.
- OpenAIEmbedder: production option (text-embedding-3-small) when
  LLM_PROVIDER=openai.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .bm25 import tokenize


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> "Embedder": ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class LSAEmbedder:
    def __init__(self, dims: int = 48, seed: int = 0):
        self.dims, self.seed = dims, seed

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)
        x = self.vectorizer.fit_transform(texts)
        n_components = min(self.dims, x.shape[0] - 1, x.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.seed).fit(x)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.svd.transform(self.vectorizer.transform(texts)))


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model

    def fit(self, texts: list[str]) -> "OpenAIEmbedder":
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode(),
            headers={"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())["data"]
        return _normalize(np.array([d["embedding"] for d in data], dtype=np.float32))
