"""Embeddings locais com sentence-transformers (carregados sob demanda)."""
from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # pesado: só quando necessário

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)
