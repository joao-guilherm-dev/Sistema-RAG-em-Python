import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit.bm25 import tokenize  # noqa: E402


class FakeEmbedder:
    """Embedding determinístico por hashing de palavras: testes rápidos, sem baixar modelo."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.calls = 0

    def encode(self, texts, show_progress=False):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in tokenize(t):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


@pytest.fixture
def embedder():
    return FakeEmbedder()
