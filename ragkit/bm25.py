"""BM25 com índice invertido e pontuação vetorizada (numpy)."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Minúsculas e sem acentos: 'Úteis' e 'uteis' viram o mesmo termo."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return _TOKEN.findall(folded)


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)

        lengths = np.zeros(self.n, dtype=np.float32)
        postings: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
        for i, doc in enumerate(docs):
            tokens = tokenize(doc)
            lengths[i] = len(tokens)
            for term, freq in Counter(tokens).items():
                ids, tfs = postings[term]
                ids.append(i)
                tfs.append(freq)

        self.doc_len = lengths
        self.avgdl = float(lengths.mean()) if self.n else 0.0
        self.postings = {
            t: (np.array(ids, dtype=np.int64), np.array(tfs, dtype=np.float32))
            for t, (ids, tfs) in postings.items()
        }
        self.idf = {
            t: math.log(1 + (self.n - ids.size + 0.5) / (ids.size + 0.5))
            for t, (ids, _) in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            ids, tfs = self.postings[term]
            out[ids] += self.idf[term] * tfs * (self.k1 + 1) / (tfs + norm[ids])
        return out
