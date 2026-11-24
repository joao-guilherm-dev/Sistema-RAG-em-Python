"""RAG em Python: ingestão, busca híbrida (denso + BM25) e geração com citações."""
from .config import Config
from .generation import RAG, Answer
from .index import Hit, Index, IndexMismatchError

__version__ = "0.2.0"
__all__ = ["Config", "Index", "IndexMismatchError", "Hit", "RAG", "Answer", "__version__"]
