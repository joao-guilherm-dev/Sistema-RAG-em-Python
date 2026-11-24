from ragkit.config import Config
from ragkit.generation import EMPTY_INDEX_MSG, RAG
from ragkit.index import Index


def test_ask_on_empty_index_does_not_call_the_llm(tmp_path, embedder):
    class ExplodingClient:
        def __getattr__(self, name):
            raise AssertionError("o LLM não deveria ser chamado com índice vazio")

    cfg = Config(index_dir=str(tmp_path / "idx"))
    rag = RAG(cfg, Index.open(cfg, embedder), client=ExplodingClient())
    ans = rag.ask("qualquer pergunta")
    assert ans.text == EMPTY_INDEX_MSG
    assert ans.sources == []
