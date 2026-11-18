"""BM25 (Okapi) lexical retrieval, implemented from scratch."""
from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = set(
    "a an the and or of to in on for is are was were be by with at from as it its this that can do does "
    "i we you they our my your if what how when who which there their than then into not no".split()
)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        # tiny stemmer: plural -> singular, good enough for policy text
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.tf = [Counter(t) for t in self.tokens]
        df = Counter(term for toks in self.tokens for term in set(toks))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.doc_len):
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                f = tf[term]
                s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out
