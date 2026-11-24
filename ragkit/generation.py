"""Geração com Claude: reescrita de perguntas de acompanhamento, respostas com citações [n]."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import Config
from .index import Hit, Index

SYSTEM_PROMPT = (
    "Você é um assistente que responde perguntas usando SOMENTE o contexto fornecido.\n"
    "Regras:\n"
    "- Cite as fontes no texto com o número entre colchetes, ex.: [1], [2].\n"
    "- Se o contexto não contiver a resposta, diga claramente que não encontrou "
    "essa informação nos documentos. Não invente.\n"
    "- Responda no mesmo idioma da pergunta, de forma objetiva."
)
CONDENSE_PROMPT = (
    "Reescreva a última pergunta do usuário como uma pergunta completa e autônoma, "
    "usando o histórico para resolver referências. Responda apenas com a pergunta reescrita."
)
EMPTY_INDEX_MSG = "O índice está vazio. Rode `python rag.py ingest <pasta>` primeiro."
NOT_FOUND_MSG = "Não encontrei nenhum trecho relevante nos documentos."
MAX_HISTORY_TURNS = 3
MAX_TURN_CHARS = 1000


@dataclass
class Answer:
    text: str
    sources: list[Hit]
    question: str  # pergunta autônoma efetivamente usada na busca


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        loc = h.name + (f", p. {h.page}" if h.page else "")
        blocks.append(f"[{i}] ({loc})\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def _text_of(response) -> str:
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()


class RAG:
    def __init__(self, cfg: Config, index: Index, client=None):
        self.cfg = cfg
        self.index = index
        self._client = client
        self.history: list[tuple[str, str]] = []  # (pergunta, resposta)

    @property
    def client(self):
        if self._client is None:
            import anthropic  # importação tardia: não precisa da API para indexar

            self._client = anthropic.Anthropic()
        return self._client

    def _condense(self, question: str) -> str:
        if not self.history:
            return question
        lines = []
        for q, a in self.history[-MAX_HISTORY_TURNS:]:
            lines.append(f"Usuário: {q[:MAX_TURN_CHARS]}")
            lines.append(f"Assistente: {a[:MAX_TURN_CHARS]}")
        convo = "\n".join(lines)
        resp = self.client.messages.create(
            model=self.cfg.llm_model,
            max_tokens=200,
            system=CONDENSE_PROMPT,
            messages=[{"role": "user", "content": f"Histórico:\n{convo}\n\nÚltima pergunta: {question}"}],
        )
        return _text_of(resp) or question

    def ask(
        self,
        question: str,
        k: int | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        if not self.index.chunks:
            return Answer(EMPTY_INDEX_MSG, [], question)

        standalone = self._condense(question)
        hits = self.index.search(standalone, k)
        if not hits:
            return Answer(NOT_FOUND_MSG, [], standalone)

        kwargs = dict(
            model=self.cfg.llm_model,
            max_tokens=self.cfg.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Contexto:\n\n{format_context(hits)}\n\nPergunta: {standalone}",
            }],
        )
        if on_text is None:
            text = _text_of(self.client.messages.create(**kwargs))
        else:
            parts: list[str] = []
            with self.client.messages.stream(**kwargs) as stream:
                for chunk in stream.text_stream:
                    on_text(chunk)
                    parts.append(chunk)
            text = "".join(parts).strip()

        self.history.append((question, text))
        return Answer(text, hits, standalone)
