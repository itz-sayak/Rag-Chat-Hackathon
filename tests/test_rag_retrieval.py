from rag import chunk_text


def test_chunk_text_basic():
    text = "word " * 1200
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    # Expect multiple chunks and overlaps
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)
