#!/usr/bin/env python3
"""
Sistema RAG completo (Retrieval-Augmented Generation) em um único arquivo.

Pipeline:
  1. Carga de documentos (.txt, .md, .pdf)
  2. Chunking por parágrafos/sentenças com sobreposição
  3. Embeddings locais (sentence-transformers, multilíngue)
  4. Índice vetorial persistido em disco (numpy) + BM25 próprio
  5. Busca híbrida (densa + BM25) com Reciprocal Rank Fusion
  6. Geração com Claude, com citações [n] e streaming
  7. Chat com reescrita de perguntas de acompanhamento

Instalação:
  pip install numpy sentence-transformers pypdf anthropic
  export ANTHROPIC_API_KEY="sua-chave"

Uso:
  python rag.py ingest ./meus_docs
  python rag.py ask "Qual é o prazo de entrega?"
  python rag.py chat
  python rag.py stats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUPPORTED = {".txt", ".md", ".pdf"}


# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    index_dir: str = os.getenv("RAG_INDEX_DIR", "rag_index")
    embed_model: str = os.getenv(
        "RAG_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    llm_model: str = os.getenv("RAG_LLM_MODEL", "claude-sonnet-5-5")
    chunk_size: int = 800        # caracteres
    chunk_overlap: int = 120     # caracteres
    top_k: int = 5               # trechos enviados ao LLM
    candidates: int = 20         # candidatos por método antes da fusão
    max_tokens: int = 1024


# --------------------------------------------------------------------------- #
# Carga de documentos
# --------------------------------------------------------------------------- #
def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_file(path: Path) -> list[tuple[str, int | None]]:
    """Retorna lista de (texto, página). Página é None para txt/md."""
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), None)]
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [
            (page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)
        ]
    return []


def discover_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED else []
    return sorted(
        p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED
    )


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # 1) quebra parágrafos grandes em pedaços <= size
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            pieces.append(para)
            continue
        buf = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            while len(sent) > size:  # sentença gigante: corte duro
                if buf:
                    pieces.append(buf)
                    buf = ""
                pieces.append(sent[:size])
                sent = sent[size:]
            if buf and len(buf) + len(sent) + 1 > size:
                pieces.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            pieces.append(buf)

    # 2) agrupa pedaços em chunks com sobreposição
    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + len(piece) + 2 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            if " " in tail:
                tail = tail[tail.find(" ") + 1:]
            cur = f"{tail}\n\n{piece}".strip()
        else:
            cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)


# --------------------------------------------------------------------------- #
# BM25 (busca lexical)
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.doc_len = np.array([len(t) for t in self.tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(docs) else 0.0
        self.tf = [Counter(t) for t in self.tokens]
        df: Counter = Counter()
        for t in self.tokens:
            df.update(set(t))
        n = len(docs)
        self.idf = {
            w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tokens), dtype=np.float32)
        q = tokenize(query)
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out


# --------------------------------------------------------------------------- #
# Índice vetorial persistente
# --------------------------------------------------------------------------- #
class Index:
    def __init__(self, cfg: Config, embedder: Embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.dir = Path(cfg.index_dir)
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.chunks: list[dict] = []
        self.files: dict[str, str] = {}  # fonte -> hash
        self._bm25: BM25 | None = None
        self.load()

    # -- persistência --
    def load(self) -> None:
        meta = self.dir / "meta.json"
        vecs = self.dir / "vectors.npy"
        if meta.exists() and vecs.exists():
            data = json.loads(meta.read_text(encoding="utf-8"))
            if data.get("embed_model") != self.cfg.embed_model:
                sys.exit(
                    f"O índice foi criado com '{data.get('embed_model')}', mas o modelo "
                    f"atual é '{self.cfg.embed_model}'. Use --reset para recriar."
                )
            self.chunks = data["chunks"]
            self.files = data["files"]
            self.vectors = np.load(vecs)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "vectors.npy", self.vectors)
        (self.dir / "meta.json").write_text(
            json.dumps(
                {
                    "embed_model": self.cfg.embed_model,
                    "files": self.files,
                    "chunks": self.chunks,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def reset(self) -> None:
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.chunks, self.files, self._bm25 = [], {}, None
        for name in ("vectors.npy", "meta.json"):
            (self.dir / name).unlink(missing_ok=True)

    # -- ingestão --
    def remove_source(self, source: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c["source"] != source]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if len(keep) else np.zeros((0, 0), np.float32)
        self.files.pop(source, None)
        self._bm25 = None

    def ingest(self, target: Path) -> None:
        files = discover_files(target)
        if not files:
            print("Nenhum arquivo suportado encontrado (.txt, .md, .pdf).")
            return
        for path in files:
            source = str(path.resolve())
            h = file_hash(path)
            if self.files.get(source) == h:
                print(f"= sem mudanças: {path.name}")
                continue
            if source in self.files:
                self.remove_source(source)

            new_chunks: list[dict] = []
            for text, page in load_file(path):
                for piece in chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap):
                    new_chunks.append(
                        {"source": source, "name": path.name, "page": page, "text": piece}
                    )
            if not new_chunks:
                print(f"! sem texto extraível: {path.name}")
                continue

            vecs = self.embedder.encode([c["text"] for c in new_chunks], show_progress=True)
            self.vectors = vecs if self.vectors.size == 0 else np.vstack([self.vectors, vecs])
            self.chunks.extend(new_chunks)
            self.files[source] = h
            self._bm25 = None
            print(f"+ indexado: {path.name} ({len(new_chunks)} trechos)")
        self.save()

    # -- busca --
    def search(self, query: str, k: int | None = None) -> list[dict]:
        if not self.chunks:
            return []
        k = k or self.cfg.top_k
        n = min(self.cfg.candidates, len(self.chunks))

        qvec = self.embedder.encode([query])[0]
        dense = self.vectors @ qvec
        dense_rank = np.argsort(-dense)[:n]

        if self._bm25 is None:
            self._bm25 = BM25([c["text"] for c in self.chunks])
        lex = self._bm25.scores(query)
        lex_rank = [i for i in np.argsort(-lex)[:n] if lex[i] > 0]

        # Reciprocal Rank Fusion
        fused: dict[int, float] = {}
        for ranking in (dense_rank, lex_rank):
            for pos, idx in enumerate(ranking):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (60 + pos + 1)

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            {**self.chunks[i], "score": fused[i], "cosine": float(dense[i])} for i in top
        ]


# --------------------------------------------------------------------------- #
# Geração
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "Você é um assistente que responde perguntas usando SOMENTE o contexto fornecido.\n"
    "Regras:\n"
    "- Cite as fontes no texto com o número entre colchetes, ex.: [1], [2].\n"
    "- Se o contexto não contiver a resposta, diga claramente que não encontrou "
    "essa informação nos documentos. Não invente.\n"
    "- Responda no mesmo idioma da pergunta, de forma objetiva."
)


def format_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        loc = f"{h['name']}" + (f", p. {h['page']}" if h["page"] else "")
        blocks.append(f"[{i}] ({loc})\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


class RAG:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.index = Index(cfg, Embedder(cfg.embed_model))
        self.history: list[dict] = []
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def _condense(self, question: str) -> str:
        """Reescreve perguntas de acompanhamento como perguntas autônomas."""
        if not self.history:
            return question
        convo = "\n".join(
            f"{'Usuário' if m['role'] == 'user' else 'Assistente'}: {m['raw']}"
            for m in self.history[-6:]
        )
        resp = self.client.messages.create(
            model=self.cfg.llm_model,
            max_tokens=200,
            system="Reescreva a última pergunta do usuário como uma pergunta completa e "
            "autônoma, usando o histórico para resolver referências. "
            "Responda apenas com a pergunta reescrita.",
            messages=[
                {
                    "role": "user",
                    "content": f"Histórico:\n{convo}\n\nÚltima pergunta: {question}",
                }
            ],
        )
        return resp.content[0].text.strip() or question

    def ask(self, question: str, k: int | None = None, stream: bool = True) -> dict:
        standalone = self._condense(question)
        hits = self.index.search(standalone, k)
        if not hits:
            msg = "O índice está vazio. Rode `python rag.py ingest <pasta>` primeiro."
            print(msg)
            return {"answer": msg, "sources": []}

        user_msg = (
            f"Contexto:\n\n{format_context(hits)}\n\n"
            f"Pergunta: {standalone}"
        )
        kwargs = dict(
            model=self.cfg.llm_model,
            max_tokens=self.cfg.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        if stream:
            parts = []
            with self.client.messages.stream(**kwargs) as s:
                for text in s.text_stream:
                    print(text, end="", flush=True)
                    parts.append(text)
            print()
            answer = "".join(parts)
        else:
            answer = self.client.messages.create(**kwargs).content[0].text

        self.history += [
            {"role": "user", "raw": question},
            {"role": "assistant", "raw": answer},
        ]
        return {"answer": answer, "sources": hits}


def print_sources(hits: list[dict]) -> None:
    if not hits:
        return
    print("\nFontes:")
    for i, h in enumerate(hits, 1):
        loc = h["name"] + (f", p. {h['page']}" if h["page"] else "")
        preview = h["text"][:90].replace("\n", " ")
        print(f"  [{i}] {loc} — \"{preview}...\"")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Sistema RAG em Python")
    parser.add_argument("--index-dir", default=None, help="Pasta do índice")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Indexa arquivos ou pastas")
    p_ing.add_argument("path")
    p_ing.add_argument("--reset", action="store_true", help="Apaga o índice antes")

    p_ask = sub.add_parser("ask", help="Faz uma pergunta")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=None, help="Nº de trechos")

    sub.add_parser("chat", help="Conversa interativa")
    sub.add_parser("stats", help="Estatísticas do índice")

    args = parser.parse_args()
    cfg = Config()
    if args.index_dir:
        cfg.index_dir = args.index_dir

    rag = RAG(cfg)

    if args.cmd == "ingest":
        if args.reset:
            rag.index.reset()
        target = Path(args.path)
        if not target.exists():
            sys.exit(f"Caminho não encontrado: {target}")
        rag.index.ingest(target)

    elif args.cmd == "ask":
        result = rag.ask(args.question, k=args.k)
        print_sources(result["sources"])

    elif args.cmd == "chat":
        print("Chat RAG — digite 'sair' para encerrar.\n")
        while True:
            try:
                q = input("Você: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"sair", "exit", "quit"}:
                break
            if not q:
                continue
            print("Assistente: ", end="")
            result = rag.ask(q)
            print_sources(result["sources"])
            print()

    elif args.cmd == "stats":
        print(f"Arquivos: {len(rag.index.files)}")
        print(f"Trechos:  {len(rag.index.chunks)}")
        print(f"Modelo de embeddings: {cfg.embed_model}")
        print(f"Modelo LLM: {cfg.llm_model}")


if __name__ == "__main__":
    main()
