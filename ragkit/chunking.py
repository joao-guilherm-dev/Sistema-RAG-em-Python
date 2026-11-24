"""Chunking por parágrafos e sentenças, com sobreposição em caracteres."""
from __future__ import annotations

import re

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_BLANKS = re.compile(r"[ \t]+")


def _split_long(text: str, size: int) -> list[str]:
    """Quebra um parágrafo em pedaços <= size, preferindo fronteiras de sentença."""
    if len(text) <= size:
        return [text]
    out: list[str] = []
    buf = ""
    for sent in _SENTENCE.split(text):
        while len(sent) > size:  # sentença maior que o limite: corte duro
            if buf:
                out.append(buf)
                buf = ""
            out.append(sent[:size])
            sent = sent[size:]
        if not sent:
            continue
        candidate = f"{buf} {sent}" if buf else sent
        if len(candidate) <= size:
            buf = candidate
        else:
            out.append(buf)
            buf = sent
    if buf:
        out.append(buf)
    return out


def _pieces(text: str, size: int) -> list[str]:
    pieces: list[str] = []
    for para in _PARAGRAPH.split(text):
        para = _BLANKS.sub(" ", para).strip()
        if para:
            pieces.extend(_split_long(para, size))
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Últimos `overlap` caracteres, sem começar no meio de uma palavra."""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    if len(tail) < len(text):
        space = tail.find(" ")
        tail = tail[space + 1:] if space != -1 else ""
    return tail.strip()


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    text = text.replace("\r\n", "\n")
    chunks: list[str] = []
    cur = ""
    for piece in _pieces(text, size):
        if not cur:
            cur = piece
        elif len(cur) + 2 + len(piece) <= size:
            cur = f"{cur}\n\n{piece}"
        else:
            chunks.append(cur)
            tail = _tail(cur, overlap)
            fits = tail and len(tail) + 2 + len(piece) <= size
            cur = f"{tail}\n\n{piece}" if fits else piece
    if cur:
        chunks.append(cur)
    return chunks
