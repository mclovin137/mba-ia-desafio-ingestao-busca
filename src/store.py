"""PGVector e manifesto do PDF ativo por collection."""
import hashlib

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from sqlalchemy import create_engine, text

DDL = """
CREATE TABLE IF NOT EXISTS rag_documents (
    collection TEXT PRIMARY KEY,
    embedding_model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    document_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL
)
"""


def make_store(settings):
    return PGVector(
        embeddings=OpenAIEmbeddings(
            api_key=settings.api_key, model=settings.embedding_model,
            dimensions=settings.dimensions,
        ),
        collection_name=settings.collection,
        connection=settings.database_url,
        embedding_length=settings.dimensions,
        distance_strategy=DistanceStrategy.COSINE,
        use_jsonb=True,
    )


def engine_for(settings):
    return create_engine(settings.database_url, pool_pre_ping=True)


def read_manifest(connection, collection):
    return connection.execute(text(
        "SELECT * FROM rag_documents WHERE collection = :collection"
    ), {"collection": collection}).mappings().first()


def check_profile(manifest, settings):
    if manifest and (manifest["embedding_model"] != settings.embedding_model
                     or manifest["dimensions"] != settings.dimensions):
        raise ValueError("Modelo/dimensão de embeddings incompatível. "
                         "Restaure a configuração ou recrie o banco e faça a ingestão.")


def chunk_id(collection, digest, index):
    return hashlib.sha256(f"{collection}:{digest}:{index}".encode()).hexdigest()
