# RAG em Python

Sistema de **Retrieval-Augmented Generation** completo em um único arquivo (`rag.py`). Você indexa seus documentos e faz perguntas em linguagem natural; as respostas são geradas pelo Claude **somente com base nos trechos recuperados**, com citação das fontes.

## Funcionalidades

- **Ingestão** de `.txt`, `.md` e `.pdf` (com número da página)
- **Reindexação incremental**: só arquivos alterados são reprocessados (hash SHA-256)
- **Chunking** por parágrafos e sentenças, com sobreposição
- **Embeddings locais** com `sentence-transformers` (modelo multilíngue, ótimo para português)
- **Índice persistido em disco** (numpy + JSON), sem banco vetorial externo
- **Busca híbrida**: similaridade semântica + BM25, combinadas por Reciprocal Rank Fusion
- **Geração com citações** `[1]`, `[2]` e resposta em streaming
- **Modo chat** com reescrita de perguntas de acompanhamento
- Responde "não encontrei" quando o contexto não tem a resposta

## Como funciona

```
Documentos ──► chunking ──► embeddings ──► índice (disco)
                                              │
Pergunta ──► (reescrita) ──► busca híbrida ◄──┘
                              (densa + BM25 + RRF)
                                    │
                             top-k trechos
                                    │
                                    ▼
                          Claude (resposta + citações)
```

## Instalação

Requer Python 3.10+.

```bash
git clone <url-do-repositorio>
cd <pasta-do-repositorio>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Defina sua chave da API da Anthropic:

```bash
export ANTHROPIC_API_KEY="sua-chave"     # Windows (PowerShell): $env:ANTHROPIC_API_KEY="sua-chave"
```

> Nunca versione sua chave. O `.gitignore` já ignora arquivos `.env`.

## Uso

```bash
# 1. Indexar um arquivo ou uma pasta inteira
python rag.py ingest ./docs

# 2. Fazer uma pergunta
python rag.py ask "Qual é o prazo de entrega?"

# 3. Conversar (com histórico)
python rag.py chat

# 4. Ver estatísticas do índice
python rag.py stats
```

Opções úteis:

| Comando | Descrição |
|---|---|
| `ingest <caminho> --reset` | Apaga o índice atual e recria do zero |
| `ask "..." -k 8` | Usa 8 trechos de contexto em vez de 5 |
| `--index-dir <pasta>` | Usa outra pasta para o índice (antes do subcomando) |

## Configuração

Por variáveis de ambiente:

| Variável | Padrão | Descrição |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Chave da API (obrigatória) |
| `RAG_LLM_MODEL` | `claude-sonnet-5-5` | Modelo usado na geração |
| `RAG_EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Modelo de embeddings |
| `RAG_INDEX_DIR` | `rag_index` | Pasta onde o índice é salvo |

Parâmetros de chunking e busca (`chunk_size`, `chunk_overlap`, `top_k`, `candidates`) ficam na classe `Config` no início do `rag.py`.

> Se trocar o modelo de embeddings, reindexe com `--reset`: vetores de modelos diferentes não são compatíveis.

## Estrutura

```
.
├── rag.py             # sistema completo + CLI
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitações e próximos passos

- O índice em numpy é ideal para até algumas dezenas de milhares de trechos. Para bases maiores, troque por FAISS, Chroma ou pgvector.
- Não há reranking com cross-encoder; adicioná-lo costuma melhorar a precisão.
- Não extrai texto de PDFs escaneados (seria necessário OCR).
- Não há suporte a `.docx`, `.html` ou outros formatos.

## Licença

Defina a licença do seu projeto (por exemplo, MIT) e adicione um arquivo `LICENSE`.
