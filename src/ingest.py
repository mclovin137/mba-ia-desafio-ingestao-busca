"""Execute: python src/ingest.py."""
import hashlib
import sys
from io import BytesIO

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy import text

from config import Settings
from store import DDL, check_profile, chunk_id, engine_for, make_store, read_manifest


def split_pdf(path):
    pages = PyPDFLoader(str(path)).load()
    pages = [page for page in pages if page.page_content.strip()]
    if not pages:
        raise ValueError("PDF sem texto extraível. PDFs digitalizados precisam de OCR.")
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, length_function=len,
    ).split_documents(pages)


def ingest_pdf(settings=None, store_factory=make_store):
    settings = settings or Settings.load()
    # Valida o arquivo antes de acessar banco ou provedor.
    payload = settings.pdf_path.read_bytes()
    PdfReader(BytesIO(payload))
    digest = hashlib.sha256(payload).hexdigest()
    engine = engine_for(settings)
    try:
        with engine.begin() as connection:
            connection.execute(text(DDL))
        with engine.begin() as connection:
            # Serializa ingestões desta collection, inclusive entre processos.
            connection.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:name, 0))"),
                               {"name": settings.collection})
            previous = read_manifest(connection, settings.collection)
            check_profile(previous, settings)
            if previous and previous["document_hash"] == digest:
                print(f"PDF já ingerido: {previous['chunk_count']} chunks. Nenhuma alteração.")
                return previous["chunk_count"]
            chunks = split_pdf(settings.pdf_path)
            # Impede publicar chunks se o arquivo mudou durante a leitura.
            if hashlib.sha256(settings.pdf_path.read_bytes()).hexdigest() != digest:
                raise ValueError("O PDF mudou durante a ingestão. Execute novamente.")
            for index, chunk in enumerate(chunks):
                chunk.metadata.update(document_sha256=digest, chunk_index=index,
                                      source=settings.pdf_path.name)
            store = store_factory(settings)
            ids = [chunk_id(settings.collection, digest, i) for i in range(len(chunks))]
            for start in range(0, len(chunks), 64):
                store.add_documents(chunks[start:start + 64], ids=ids[start:start + 64])
            connection.execute(text("""
                INSERT INTO rag_documents
                    (collection, embedding_model, dimensions, document_hash, chunk_count)
                VALUES (:collection, :model, :dimensions, :digest, :count)
                ON CONFLICT (collection) DO UPDATE SET
                    embedding_model = EXCLUDED.embedding_model,
                    dimensions = EXCLUDED.dimensions,
                    document_hash = EXCLUDED.document_hash,
                    chunk_count = EXCLUDED.chunk_count
            """), dict(collection=settings.collection, model=settings.embedding_model,
                       dimensions=settings.dimensions, digest=digest, count=len(chunks)))
        # Só publica o manifesto depois de todos os lotes terem sido persistidos.
        print(f"Ingestão concluída: {len(chunks)} chunks.")
        return len(chunks)
    finally:
        engine.dispose()


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as error:
        print(f"Erro na ingestão: {type(error).__name__}. {error if isinstance(error, (ValueError, FileNotFoundError)) else 'Verifique banco, credenciais e conectividade.'}", file=sys.stderr)
        sys.exit(1)
