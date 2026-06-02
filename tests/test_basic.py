from app.config import CHUNK_SIZE, OVERLAP_RATIO, TOP_K
from scripts.ingest import chunk_text


def test_stats_values_are_valid():
    assert isinstance(CHUNK_SIZE, int)
    assert CHUNK_SIZE <= 1024

    assert isinstance(OVERLAP_RATIO, float)
    assert 0 <= OVERLAP_RATIO <= 0.3

    assert isinstance(TOP_K, int)
    assert 1 <= TOP_K <= 30


def test_chunk_text_basic():
    text = " ".join([f"word{i}" for i in range(1000)])
    chunks = chunk_text(text, chunk_size=100, overlap_ratio=0.2)

    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 100 for chunk in chunks)
