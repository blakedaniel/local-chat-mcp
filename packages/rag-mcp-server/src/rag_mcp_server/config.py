"""Configuration settings for the RAG MCP Server."""

import os
from dataclasses import dataclass
import dotenv

dotenv.load_dotenv()

@dataclass
class Settings:
    """RAG MCP Server configuration with environment variable overrides."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8001

    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5-coder:1.5b-base"
    embedding_model: str = "nomic-embed-text"
    request_timeout: float = 120.0

    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "rag_documents"

    # RAG settings
    chunk_size: int = 1024
    chunk_overlap: int = 200
    similarity_top_k: int = 3

    # Data settings
    data_dir: str = "./data"

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            host=os.getenv("RAG_HOST", cls.host),
            port=int(os.getenv("RAG_PORT", cls.port)),
            ollama_url=os.getenv("RAG_OLLAMA_URL", cls.ollama_url),
            llm_model=os.getenv("RAG_LLM_MODEL", cls.llm_model),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", cls.embedding_model),
            request_timeout=float(os.getenv("RAG_REQUEST_TIMEOUT", cls.request_timeout)),
            qdrant_host=os.getenv("RAG_QDRANT_HOST", cls.qdrant_host),
            qdrant_port=int(os.getenv("RAG_QDRANT_PORT", cls.qdrant_port)),
            collection_name=os.getenv("RAG_COLLECTION_NAME", cls.collection_name),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", cls.chunk_overlap)),
            similarity_top_k=int(os.getenv("RAG_SIMILARITY_TOP_K", cls.similarity_top_k)),
            data_dir=os.getenv("RAG_DATA_DIR", cls.data_dir),
        )


# Global settings instance
settings = Settings.from_env()
