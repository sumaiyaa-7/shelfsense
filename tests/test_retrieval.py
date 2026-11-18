import pytest

from shelfsense.config import get_settings
from shelfsense.retrieval.chunker import load_corpus
from shelfsense.retrieval.retriever import MODES, Retriever


@pytest.fixture(scope="module")
def retriever():
    return Retriever.from_dir(get_settings().policies_dir)


def test_chunks_have_sections():
    chunks = load_corpus(get_settings().policies_dir)
    assert len(chunks) >= 20
    assert all(c.section and c.text for c in chunks)


@pytest.mark.parametrize("mode", MODES)
def test_finds_restocking_fee(retriever, mode):
    hits = retriever.search("restocking fee on an opened printer", k=3, mode=mode)
    assert any(h.chunk.section == "Electronics and Technology" for h in hits)


def test_bad_mode(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")
