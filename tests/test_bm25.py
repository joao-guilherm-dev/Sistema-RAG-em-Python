from ragkit.bm25 import BM25, tokenize


def test_tokenize_folds_case_and_accents():
    assert tokenize("Prazo de Entrega: ÚTEIS, 30 dias") == ["prazo", "de", "entrega", "uteis", "30", "dias"]


def test_bm25_prefers_documents_with_query_terms():
    docs = [
        "o gato dorme no sofá",
        "a entrega leva trinta dias úteis",
        "o cachorro corre no parque",
    ]
    scores = BM25(docs).scores("entrega úteis")
    assert scores.argmax() == 1
    assert scores[0] == 0 and scores[2] == 0


def test_bm25_empty_corpus():
    assert BM25([]).scores("qualquer coisa").size == 0
