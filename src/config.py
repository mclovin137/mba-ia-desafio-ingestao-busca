"""Configuração compartilhada; caminhos relativos à raiz do projeto."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    database_url: str
    collection: str
    api_key: str
    embedding_model: str
    dimensions: int
    chat_model: str
    pdf_path: Path

    @classmethod
    def load(cls):
        load_dotenv(ROOT / ".env")
        def value(name, default=""):
            result = os.getenv(name, default).strip()
            if not result:
                raise ValueError(f"Configure {name} no arquivo .env.")
            return result

        dimensions = int(value("OPENAI_EMBEDDING_DIMENSIONS", "1536"))
        if dimensions <= 0:
            raise ValueError("OPENAI_EMBEDDING_DIMENSIONS deve ser positivo.")
        pdf = Path(value("PDF_PATH", "document.pdf"))
        return cls(
            value("DATABASE_URL"), value("PG_VECTOR_COLLECTION_NAME", "document_pdf"),
            value("OPENAI_API_KEY"), value("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            dimensions, value("OPENAI_CHAT_MODEL", "gpt-5-mini"),
            pdf if pdf.is_absolute() else ROOT / pdf,
        )
