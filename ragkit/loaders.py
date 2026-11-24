"""Descoberta e leitura dos arquivos de origem."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class Page:
    text: str
    number: int | None  # None para txt/md


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    # Windows costuma salvar em cp1252; tenta UTF-8 primeiro e não descarta caracteres em silêncio
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_file(path: Path) -> list[Page]:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return [Page(_read_text(path), None)]
    if ext == ".pdf":
        from pypdf import PdfReader  # importação tardia: só é necessária para PDFs

        reader = PdfReader(str(path))
        return [Page(p.extract_text() or "", i) for i, p in enumerate(reader.pages, 1)]
    raise ValueError(f"Formato não suportado: {ext}")


def discover_files(target: Path) -> list[Path]:
    """Lista arquivos suportados, ignorando pastas ocultas (.git, .venv, ...)."""
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED else []
    found = []
    for p in target.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED:
            continue
        if any(part.startswith(".") for part in p.relative_to(target).parts):
            continue
        found.append(p)
    return sorted(found)
