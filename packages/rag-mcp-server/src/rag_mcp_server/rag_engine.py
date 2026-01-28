"""RAG Engine - Wraps LlamaIndex functionality for document indexing and querying."""

import os
from typing import Optional

from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from .config import settings as config


class RAGEngine:
    """RAG Engine that manages document indexing and querying with LlamaIndex."""

    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.vector_store: Optional[QdrantVectorStore] = None
        self.index: Optional[VectorStoreIndex] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the RAG engine with LlamaIndex, Ollama, and Qdrant."""
        if self._initialized:
            return

        # Configure LlamaIndex global settings
        Settings.llm = Ollama(
            model=config.llm_model,
            base_url=config.ollama_url,
            request_timeout=config.request_timeout,
        )

        Settings.embed_model = OllamaEmbedding(
            model_name=config.embedding_model,
            base_url=config.ollama_url,
        )

        Settings.chunk_size = config.chunk_size
        Settings.chunk_overlap = config.chunk_overlap

        # Connect to Qdrant
        self.client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)

        # Create vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=config.collection_name,
        )

        # Load existing index if collection exists
        if self._collection_exists():
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)

        self._initialized = True

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.client:
            self.client.close()
        self._initialized = False

    def _collection_exists(self) -> bool:
        """Check if the Qdrant collection already exists."""
        if not self.client:
            return False
        try:
            collections = self.client.get_collections().collections
            return any(c.name == config.collection_name for c in collections)
        except Exception:
            return False

    def _ensure_data_dir(self) -> None:
        """Ensure the data directory exists."""
        if not os.path.exists(config.data_dir):
            os.makedirs(config.data_dir)

    async def query(self, question: str, top_k: Optional[int] = None) -> dict:
        """Query the RAG system with a natural language question."""
        if not self.index:
            return {
                "answer": "No documents have been indexed yet. Please ingest some documents first.",
                "sources": [],
            }

        k = top_k or config.similarity_top_k
        query_engine = self.index.as_query_engine(similarity_top_k=k)

        response = query_engine.query(question)

        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                source_info = {
                    "file_name": node.node.metadata.get("file_name", "Unknown"),
                    "relevance": round(node.score, 3) if hasattr(node, "score") and node.score else 0.0,
                    "preview": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                }
                sources.append(source_info)

        return {
            "answer": str(response),
            "sources": sources,
        }

    async def ingest_document(self, content: str, filename: str) -> dict:
        """Ingest a single document into the RAG system."""
        doc = Document(text=content, metadata={"file_name": filename})

        if self.index is None:
            # Create new index with this document
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex.from_documents(
                [doc],
                storage_context=storage_context,
                show_progress=True,
            )
        else:
            # Insert into existing index
            self.index.insert(doc)

        return {
            "status": "success",
            "document_id": doc.doc_id,
            "filename": filename,
        }

    async def ingest_from_path(self, path: str) -> dict:
        """Ingest documents from a file or directory path."""
        if not os.path.exists(path):
            return {
                "status": "error",
                "message": f"Path does not exist: {path}",
                "documents_ingested": 0,
                "files": [],
            }

        # Load documents using SimpleDirectoryReader
        if os.path.isfile(path):
            # Single file
            from llama_index.core import Document as LIDocument

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            filename = os.path.basename(path)
            documents = [LIDocument(text=content, metadata={"file_name": filename})]
            files = [filename]
        else:
            # Directory
            reader = SimpleDirectoryReader(path)
            documents = reader.load_data()
            files = [doc.metadata.get("file_name", "Unknown") for doc in documents]

        if not documents:
            return {
                "status": "error",
                "message": "No documents found at the specified path",
                "documents_ingested": 0,
                "files": [],
            }

        # Build or update index
        if self.index is None:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True,
            )
        else:
            for doc in documents:
                self.index.insert(doc)

        return {
            "status": "success",
            "documents_ingested": len(documents),
            "files": list(set(files)),
        }

    async def rebuild_index(self) -> dict:
        """Rebuild the entire index from the data directory."""
        self._ensure_data_dir()

        # Check for documents in data directory
        files = [f for f in os.listdir(config.data_dir) if not f.startswith(".")]
        if not files:
            return {
                "status": "error",
                "message": f"No documents found in {config.data_dir}",
                "documents_indexed": 0,
            }

        # Delete existing collection
        if self._collection_exists():
            self.client.delete_collection(config.collection_name)

        # Recreate vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=config.collection_name,
        )

        # Load and index all documents
        reader = SimpleDirectoryReader(config.data_dir)
        documents = reader.load_data()

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )

        return {
            "status": "success",
            "documents_indexed": len(documents),
        }

    async def list_documents(self) -> dict:
        """List all documents in the index."""
        if not self._collection_exists():
            return {"documents": [], "total": 0}

        try:
            # Get collection info
            collection_info = self.client.get_collection(config.collection_name)
            points_count = collection_info.points_count

            # Scroll through points to get document metadata
            documents = []
            seen_files = set()

            # Use scroll to get all points
            offset = None
            while True:
                result = self.client.scroll(
                    collection_name=config.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                )
                points, next_offset = result

                for point in points:
                    if point.payload:
                        file_name = point.payload.get("file_name", "Unknown")
                        if file_name not in seen_files:
                            seen_files.add(file_name)
                            documents.append({
                                "id": str(point.id),
                                "filename": file_name,
                            })

                if next_offset is None:
                    break
                offset = next_offset

            return {
                "documents": documents,
                "total": len(documents),
                "chunks": points_count,
            }
        except UnexpectedResponse:
            return {"documents": [], "total": 0}

    async def delete_document(self, filename: str) -> dict:
        """Delete a document from the index by filename."""
        if not self._collection_exists():
            return {
                "status": "error",
                "message": "No index exists",
            }

        try:
            # Delete points with matching filename
            self.client.delete(
                collection_name=config.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {"key": "file_name", "match": {"value": filename}}
                        ]
                    }
                },
            )

            return {
                "status": "success",
                "message": f"Deleted document: {filename}",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }

    async def get_stats(self) -> dict:
        """Get statistics about the RAG index."""
        if not self._collection_exists():
            return {
                "status": "no_index",
                "document_count": 0,
                "chunk_count": 0,
                "collection_name": config.collection_name,
                "llm_model": config.llm_model,
                "embedding_model": config.embedding_model,
            }

        try:
            collection_info = self.client.get_collection(config.collection_name)
            return {
                "status": "ready",
                "chunk_count": collection_info.points_count,
                "collection_name": config.collection_name,
                "llm_model": config.llm_model,
                "embedding_model": config.embedding_model,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    async def check_health(self) -> dict:
        """Check the health of all dependencies."""
        health = {
            "qdrant": "unknown",
            "ollama": "unknown",
            "index": "unknown",
        }

        # Check Qdrant
        try:
            self.client.get_collections()
            health["qdrant"] = "connected"
        except Exception as e:
            health["qdrant"] = f"error: {str(e)}"

        # Check Ollama (basic connectivity)
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{config.ollama_url}/api/tags", timeout=5.0)
                if resp.status_code == 200:
                    health["ollama"] = "connected"
                else:
                    health["ollama"] = f"error: status {resp.status_code}"
        except Exception as e:
            health["ollama"] = f"error: {str(e)}"

        # Check index
        if self.index is not None:
            health["index"] = "ready"
        elif self._collection_exists():
            health["index"] = "collection_exists"
        else:
            health["index"] = "no_index"

        return health
