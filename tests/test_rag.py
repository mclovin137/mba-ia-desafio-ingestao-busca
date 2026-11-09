import os
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_postgres import PGVector
from pypdf import PdfWriter
from sqlalchemy import text

import chat
import search
from config import ROOT, Settings
from ingest import ingest_pdf, split_pdf
from store import check_profile, engine_for


def test_pdf_is_split_with_page_metadata():
    chunks = split_pdf(ROOT / "document.pdf")
    assert chunks
    assert all(0 < len(chunk.page_content) <= 1000 for chunk in chunks)
    assert all("page" in chunk.metadata for chunk in chunks)


def test_scanned_pdf_fails_before_embedding(tmp_path):
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(path)
    with pytest.raises(ValueError, match="OCR"):
        split_pdf(path)


def test_retrieval_uses_top_ten_and_active_pdf():
    store = Mock()
    store.similarity_search_with_score.return_value = [
        (Document(page_content="Receita de 10 milhões.", metadata={"page": 2}), 0.1)
    ]
    context = search.retrieve_context(store, "Qual a receita?", "active-hash")
    store.similarity_search_with_score.assert_called_once_with(
        query="Qual a receita?", k=10, filter={"document_sha256": "active-hash"}
    )
    assert "página 3" in context and "10 milhões" in context


@pytest.mark.parametrize("model,dimensions", [("other", 1536), ("same", 768)])
def test_embedding_mismatch_is_rejected(model, dimensions):
    settings = Mock(embedding_model="same", dimensions=1536)
    with pytest.raises(ValueError, match="incompatível"):
        check_profile({"embedding_model": model, "dimensions": dimensions}, settings)


def test_cli_blank_question_and_exit(monkeypatch, capsys):
    answers = Mock(return_value="Resposta do PDF")
    monkeypatch.setattr(chat, "search_prompt", lambda: answers)
    inputs = iter([" ", "Pergunta", "sair"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert chat.main() == 0
    answers.assert_called_once_with("Pergunta")
    assert "RESPOSTA: Resposta do PDF" in capsys.readouterr().out


def test_settings_resolve_pdf_from_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://unused")
    monkeypatch.setenv("PDF_PATH", "document.pdf")
    assert Settings.load().pdf_path == ROOT / "document.pdf"


@pytest.mark.integration
def test_real_pgvector_ingestion_search_and_failed_replacement(monkeypatch, tmp_path):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Defina TEST_DATABASE_URL para testar PostgreSQL + pgVector.")
    settings = Settings(url, "test_" + uuid.uuid4().hex, "test-key", "fake-8", 8,
                        "fake-chat", ROOT / "document.pdf")
    stores = []
    def factory(config):
        store = PGVector(embeddings=DeterministicFakeEmbedding(size=8),
                         connection=config.database_url, collection_name=config.collection,
                         embedding_length=8, use_jsonb=True)
        stores.append(store)
        return store
    engine = engine_for(settings)
    try:
        count = ingest_pdf(settings, factory)
        assert ingest_pdf(settings, factory) == count
        with engine.connect() as connection:
            actual = connection.execute(text("""
                SELECT count(*) FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON c.uuid=e.collection_id
                WHERE c.name=:name
            """), {"name": settings.collection}).scalar_one()
            assert actual == count
        calls = []
        def respond(prompt):
            calls.append(prompt.to_messages())
            return AIMessage(content="Resposta controlada")
        answer = search.search_prompt(settings=settings, store_factory=factory,
                                     llm=RunnableLambda(respond))
        assert answer("Qual o assunto?") == "Resposta controlada"
        assert "CONTEXTO:" in calls[0][1].content
        # A new snapshot with no hits must not call the model.
        monkeypatch.setattr(stores[-1], "similarity_search_with_score", lambda **_: [])
        assert answer("Sem contexto") == search.FALLBACK
        assert len(calls) == 1
        # Failed replacement preserves the previously published document.
        replacement = tmp_path / "replacement.pdf"
        replacement.write_bytes(settings.pdf_path.read_bytes() + b"\n% changed")
        broken = Mock()
        broken.add_documents.side_effect = RuntimeError("provider unavailable")
        with engine.connect() as connection:
            old_hash = connection.execute(text(
                "SELECT document_hash FROM rag_documents WHERE collection=:name"
            ), {"name": settings.collection}).scalar_one()
        with pytest.raises(RuntimeError):
            ingest_pdf(replace(settings, pdf_path=replacement), lambda _: broken)
        with engine.connect() as connection:
            assert connection.execute(text(
                "SELECT document_hash FROM rag_documents WHERE collection=:name"
            ), {"name": settings.collection}).scalar_one() == old_hash
    finally:
        if stores:
            stores[0].delete_collection()
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM rag_documents WHERE collection=:name"),
                               {"name": settings.collection})
        engine.dispose()
