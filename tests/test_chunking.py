from ragkit.chunking import chunk_text


def test_chunks_respect_size():
    text = " ".join(f"Frase número {i} sobre um assunto qualquer." for i in range(300))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_consecutive_chunks_overlap():
    text = " ".join(f"Frase {i} termina aqui." for i in range(200))
    chunks = chunk_text(text, size=200, overlap=60)
    assert len(chunks) > 2
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()) & set(b.split())


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_windows_line_endings_are_handled():
    text = "Primeiro parágrafo.\r\n\r\nSegundo parágrafo."
    assert chunk_text(text, size=800, overlap=0) == ["Primeiro parágrafo.\n\nSegundo parágrafo."]


def test_very_long_word_is_hard_cut():
    chunks = chunk_text("x" * 1000, size=300, overlap=0)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(chunks) == "x" * 1000
