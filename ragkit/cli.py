"""Interface de linha de comando: ingest, ask, chat, stats."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .embeddings import Embedder
from .generation import RAG, Answer
from .index import Hit, Index, IndexMismatchError, IngestReport


def _print_report(r: IngestReport) -> None:
    for name, n in r.indexed:
        print(f"+ indexado: {name} ({n} trechos)")
    for name in r.unchanged:
        print(f"= sem mudanças: {name}")
    for name, why in r.failed:
        print(f"! falhou: {name} ({why})")
    for name in r.removed:
        print(f"- removido do índice: {name}")
    if not (r.indexed or r.unchanged or r.failed):
        print("Nenhum arquivo suportado (.txt, .md, .pdf) encontrado.")


def _print_sources(hits: list[Hit]) -> None:
    if not hits:
        return
    print("\nFontes:")
    for i, h in enumerate(hits, 1):
        loc = h.name + (f", p. {h.page}" if h.page else "")
        preview = " ".join(h.text[:90].split())
        print(f'  [{i}] {loc} — "{preview}..."')


def _answer(rag: RAG, question: str, k: int | None = None) -> Answer:
    streamed = False

    def on_text(chunk: str) -> None:
        nonlocal streamed
        streamed = True
        print(chunk, end="", flush=True)

    ans = rag.ask(question, k=k, on_text=on_text)
    if streamed:
        print()
    else:
        print(ans.text)
    _print_sources(ans.sources)
    return ans


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sistema RAG em Python")
    parser.add_argument("--index-dir", default=None, help="Pasta do índice")
    parser.add_argument("-v", "--verbose", action="store_true", help="Logs detalhados")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Indexa arquivos ou pastas")
    p_ing.add_argument("path")
    p_ing.add_argument("--reset", action="store_true", help="Apaga o índice antes")

    p_ask = sub.add_parser("ask", help="Faz uma pergunta")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=None, help="Nº de trechos")

    sub.add_parser("chat", help="Conversa interativa")
    sub.add_parser("stats", help="Estatísticas do índice")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    cfg = Config.from_env(args.index_dir)
    embedder = Embedder(cfg.embed_model, cfg.embed_batch_size)  # modelo só carrega quando usado

    try:
        if args.cmd == "ingest":
            target = Path(args.path)
            if not target.exists():
                sys.exit(f"Caminho não encontrado: {target}")
            if args.reset:
                Index.wipe(cfg.index_dir)
            _print_report(Index.open(cfg, embedder).ingest(target))
            return

        index = Index.open(cfg, embedder)

        if args.cmd == "stats":
            for key, value in index.stats().items():
                print(f"{key}: {value}")
            return

        rag = RAG(cfg, index)
        if args.cmd == "ask":
            _answer(rag, args.question, k=args.k)
        elif args.cmd == "chat":
            print("Chat RAG — digite 'sair' para encerrar.\n")
            while True:
                try:
                    q = input("Você: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if q.lower() in {"sair", "exit", "quit"}:
                    break
                if q:
                    print("Assistente: ", end="")
                    _answer(rag, q)
                    print()
    except IndexMismatchError as exc:
        sys.exit(str(exc))
