"""Configuração centralizada. Valores podem vir de variáveis de ambiente via Config.from_env()."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    index_dir: str = "rag_index"
    embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "claude-sonnet-5-5"
    chunk_size: int = 800        # caracteres
    chunk_overlap: int = 120     # caracteres
    top_k: int = 5               # trechos enviados ao LLM
    candidates: int = 20         # candidatos por método antes da fusão
    rrf_k: int = 60              # constante do Reciprocal Rank Fusion
    embed_batch_size: int = 32
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap deve ser menor que chunk_size (e não negativo)")
        if self.top_k < 1 or self.candidates < 1:
            raise ValueError("top_k e candidates devem ser >= 1")

    @classmethod
    def from_env(cls, index_dir: str | None = None) -> "Config":
        return cls(
            index_dir=index_dir or os.getenv("RAG_INDEX_DIR", cls.index_dir),
            embed_model=os.getenv("RAG_EMBED_MODEL", cls.embed_model),
            llm_model=os.getenv("RAG_LLM_MODEL", cls.llm_model),
        )

    def index_settings(self) -> dict:
        """Parâmetros que, se mudarem, invalidam o índice já salvo."""
        return {
            "embed_model": self.embed_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
