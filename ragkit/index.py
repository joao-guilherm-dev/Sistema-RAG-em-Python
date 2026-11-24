"""Índice persistido em disco (numpy + JSON) com ingestão incremental e busca híbrida."""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .bm25 import BM25
from .chunking import chunk_text
from .config import Config
from .embeddings import Embedder
from .loaders import discover_files, file_hash, load_file

log = logging.getLogger(__name__)

VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"


class IndexMismatchError(RuntimeError):
    """Índice em disco foi criado com outra configuração (modelo, chunking) ou está corrompido."""


@dataclass
class Hit:
    source: str
    name: str
    page: int | None
    text: str
    score: float
    cosine: float


@dataclass
class IngestReport:
    indexed: list[tuple[str, int]] = field(default_factory=list)   # (arquivo, nº de trechos)
    unchanged: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)    # (arquivo, motivo)
    removed: list[str] = field(default_factory=list)


def _atomic_write(path: Path, writer: Callable[[object], None]) -> None:
    """Escreve em arquivo temporário e troca de uma vez: evita índice pela metade se o processo cair."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        writer(f)
    os.replace(tmp, path)


class Index:
    def __init__(self, cfg: Config, embedder: Embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.dir = Path(cfg.index_dir)
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.chunks: list[dict] = []
        self.files: dict[str, str] = {}  # caminho absoluto -> sha256
        self._bm25: BM25 | None = None

    @classmethod
    def open(cls, cfg: Config, embedder: Embedder) -> "Index":
        index = cls(cfg, embedder)
        index._load()
        return index

    @staticmethod
    def wipe(index_dir: str | Path) -> None:
        for name in (VECTORS_FILE, META_FILE):
            (Path(index_dir) / name).unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Persistência
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        meta_path, vec_path = self.dir / META_FILE, self.dir / VECTORS_FILE
        if not (meta_path.exists() and vec_path.exists()):
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stored, current = meta.get("settings", {}), self.cfg.index_settings()
        if stored != current:
            raise IndexMismatchError(
                f"O índice em '{self.dir}' foi criado com {stored}, mas a configuração "
                f"atual é {current}. Rode `ingest --reset` para recriar."
            )
        self.vectors = np.load(vec_path)
        self.chunks = meta["chunks"]
        self.files = meta["files"]
        if len(self.chunks) != len(self.vectors):
            raise IndexMismatchError("Índice inconsistente (vetores e trechos não batem). Rode `ingest --reset`.")

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.dir / VECTORS_FILE, lambda f: np.save(f, self.vectors))
        meta = {"settings": self.cfg.index_settings(), "files": self.files, "chunks": self.chunks}
        payload = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        _atomic_write(self.dir / META_FILE, lambda f: f.write(payload))

    # ------------------------------------------------------------------ #
    # Mutações
    # ------------------------------------------------------------------ #
    def _drop_source(self, source: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c["source"] != source]
        if len(keep) == len(self.chunks):
            return
        self.vectors = self.vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        self.chunks = [self.chunks[i] for i in keep]
        self.files.pop(source, None)
        self._bm25 = None

    def _add(self, chunks: list[dict], vecs: np.ndarray, source: str, digest: str) -> None:
        self.vectors = vecs if self.vectors.size == 0 else np.vstack([self.vectors, vecs])
        self.chunks.extend(chunks)
        self.files[source] = digest
        self._bm25 = None

    def _build_chunks(self, path: Path, source: str) -> list[dict]:
        out = []
        for page in load_file(path):
            for piece in chunk_text(page.text, self.cfg.chunk_size, self.cfg.chunk_overlap):
                out.append({"source": source, "name": path.name, "page": page.number, "text": piece})
        return out

    def ingest(self, target: Path) -> IngestReport:
        target = target.resolve()
        report = IngestReport()
        seen: set[str] = set()
        changed = False
        try:
            for path in discover_files(target):
                source = str(path.resolve())
                seen.add(source)
                digest = file_hash(path)
                if self.files.get(source) == digest:
                    report.unchanged.append(path.name)
                    continue
                try:
                    chunks = self._build_chunks(path, source)
                except Exception as exc:  # PDF corrompido, permissão negada, etc.
                    log.warning("Falha ao ler %s: %s", path, exc)
                    report.failed.append((path.name, str(exc)))
                    continue
                if not chunks:
                    report.failed.append((path.name, "sem texto extraível"))
                    continue
                self._drop_source(source)  # remove versão antiga do arquivo, se existir
                vecs = self.embedder.encode([c["text"] for c in chunks], show_progress=True)
                self._add(chunks, vecs, source, digest)
                report.indexed.append((path.name, len(chunks)))
                changed = True

            # arquivos apagados da pasta ingerida saem do índice
            if target.is_dir():
                for source in list(self.files):
                    if source not in seen and Path(source).is_relative_to(target):
                        report.removed.append(Path(source).name)
                        self._drop_source(source)
                        changed = True
        finally:
            if changed:
                self.save()  # salva mesmo se a ingestão for interrompida no meio
        return report

    # ------------------------------------------------------------------ #
    # Busca
    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        k = k or self.cfg.top_k
        n = min(self.cfg.candidates, len(self.chunks))

        dense = self.vectors @ self.embedder.encode([query])[0]
        dense_rank = np.argsort(-dense)[:n].tolist()

        if self._bm25 is None:
            self._bm25 = BM25([c["text"] for c in self.chunks])
        lex = self._bm25.scores(query)
        lex_rank = [int(i) for i in np.argsort(-lex)[:n] if lex[i] > 0]

        fused: dict[int, float] = defaultdict(float)  # Reciprocal Rank Fusion
        for ranking in (dense_rank, lex_rank):
            for pos, idx in enumerate(ranking):
                fused[idx] += 1.0 / (self.cfg.rrf_k + pos + 1)

        top = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
        hits = []
        for i in top:
            c = self.chunks[i]
            hits.append(
                Hit(
                    source=c["source"], name=c["name"], page=c["page"], text=c["text"],
                    score=fused[i], cosine=float(dense[i]),
                )
            )
        return hits

    def stats(self) -> dict:
        return {"arquivos": len(self.files), "trechos": len(self.chunks), **self.cfg.index_settings()}
