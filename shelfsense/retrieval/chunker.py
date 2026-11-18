"""Split markdown policy documents into section-aware chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc: str        # file stem, e.g. "returns_policy"
    title: str      # document H1
    section: str    # H2 the chunk belongs to
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc} > {self.section}"


def chunk_markdown(path: Path, max_words: int = 90, overlap: int = 20) -> list[Chunk]:
    """One chunk per H2 section; long sections are split into overlapping word windows.

    Section-aware chunking keeps each chunk about a single topic, which helps
    retrieval precision and gives clean, human-readable citations.
    """
    text = path.read_text()
    title_match = re.search(r"^# (.+)$", text, flags=re.M)
    title = title_match.group(1).strip() if title_match else path.stem
    chunks: list[Chunk] = []
    for i, block in enumerate(re.split(r"^## ", text, flags=re.M)[1:]):
        section, _, body = block.partition("\n")
        words = body.split()
        step = max_words - overlap
        for j, start in enumerate(range(0, max(len(words), 1), step)):
            window = " ".join(words[start:start + max_words])
            if not window:
                continue
            chunks.append(Chunk(f"{path.stem}-{i}-{j}", path.stem, title, section.strip(), window))
            if start + max_words >= len(words):
                break
    return chunks


def load_corpus(policies_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(policies_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path))
    return chunks
