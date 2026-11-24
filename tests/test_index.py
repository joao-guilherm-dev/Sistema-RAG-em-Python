from pathlib import Path

import pytest

from ragkit.config import Config
from ragkit.index import Index, IndexMismatchError


@pytest.fixture
def cfg(tmp_path):
    return Config(index_dir=str(tmp_path / "idx"), chunk_size=300, chunk_overlap=40)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_ingest_is_incremental(tmp_path, cfg, embedder):
    docs = tmp_path / "docs"
    write(docs / "a.md", "Prazo de entrega: 30 dias úteis.")
    write(docs / "b.txt", "Política de devolução: 7 dias.")

    r1 = Index(cfg, embedder).ingest(docs)
    assert sorted(n for n, _ in r1.indexed) == ["a.md", "b.txt"]

    r2 = Index.open(cfg, embedder).ingest(docs)
    assert sorted(r2.unchanged) == ["a.md", "b.txt"]
    assert not r2.indexed

    write(docs / "a.md", "Prazo de entrega: 15 dias corridos.")
    r3 = Index.open(cfg, embedder).ingest(docs)
    assert [n for n, _ in r3.indexed] == ["a.md"]
    assert r3.unchanged == ["b.txt"]


def test_deleted_files_are_pruned(tmp_path, cfg, embedder):
    docs = tmp_path / "docs"
    write(docs / "a.md", "Conteúdo A.")
    write(docs / "b.txt", "Conteúdo B.")
    Index(cfg, embedder).ingest(docs)

    (docs / "b.txt").unlink()
    index = Index.open(cfg, embedder)
    report = index.ingest(docs)
    assert report.removed == ["b.txt"]
    assert {c["name"] for c in index.chunks} == {"a.md"}


def test_files_outside_ingested_folder_are_kept(tmp_path, cfg, embedder):
    write(tmp_path / "outra" / "x.md", "Arquivo de outra pasta.")
    write(tmp_path / "docs" / "a.md", "Arquivo A.")
    index = Index(cfg, embedder)
    index.ingest(tmp_path / "outra")
    index.ingest(tmp_path / "docs")
    assert {c["name"] for c in Index.open(cfg, embedder).chunks} == {"x.md", "a.md"}


def test_search_finds_relevant_document(tmp_path, cfg, embedder):
    docs = tmp_path / "docs"
    write(docs / "rh.md", "Férias: o colaborador tem direito a 30 dias após 12 meses de trabalho.")
    write(docs / "ti.md", "Senhas devem ter no mínimo 12 caracteres e ser trocadas a cada 90 dias.")
    index = Index(cfg, embedder)
    index.ingest(docs)

    hits = index.search("caracteres mínimos senha", k=1)
    assert hits[0].name == "ti.md"


def test_config_change_requires_reset(tmp_path, cfg, embedder):
    write(tmp_path / "docs" / "a.md", "Algum texto.")
    Index(cfg, embedder).ingest(tmp_path / "docs")

    changed = Config(index_dir=cfg.index_dir, chunk_size=500, chunk_overlap=40)
    with pytest.raises(IndexMismatchError):
        Index.open(changed, embedder)

    Index.wipe(cfg.index_dir)
    assert Index.open(changed, embedder).chunks == []


def test_empty_index_search_returns_nothing(cfg, embedder):
    assert Index.open(cfg, embedder).search("qualquer") == []
