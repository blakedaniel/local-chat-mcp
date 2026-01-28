"""RAG MCP Server - Exposes LlamaIndex RAG functionality via MCP protocol."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp.server.fastmcp import FastMCP

from .config import settings
from .rag_engine import RAGEngine
from .tools import register_tools


# Create the RAG engine instance
rag_engine = RAGEngine()


@asynccontextmanager
async def lifespan(mcp: FastMCP) -> AsyncIterator[dict]:
    """Initialize RAG engine on startup, cleanup on shutdown."""
    print(f"Initializing RAG MCP Server...")
    print(f"  LLM Model: {settings.llm_model}")
    print(f"  Embedding Model: {settings.embedding_model}")
    print(f"  Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"  Collection: {settings.collection_name}")

    await rag_engine.initialize()
    print("RAG Engine initialized successfully!")

    yield {"rag_engine": rag_engine}

    print("Shutting down RAG MCP Server...")
    await rag_engine.shutdown()


# Create the FastMCP server
mcp = FastMCP(
    name="rag-server",
    instructions="""RAG (Retrieval-Augmented Generation) server for document Q&A.

This server allows you to:
- Ingest documents (text, PDF, markdown, etc.) into a vector database
- Query the documents using natural language questions
- Get answers with source attribution

Available tools:
- query: Ask questions about indexed documents
- ingest_document: Add a document by providing its content
- ingest_from_path: Index documents from a file or directory path
- rebuild_index: Rebuild the entire index from the data directory
- list_documents: List all indexed documents
- delete_document: Remove a document from the index
- get_rag_stats: Get index statistics
- check_rag_health: Check health of all components

Powered by LlamaIndex, Ollama, and Qdrant.""",
    host=settings.host,
    port=settings.port,
    lifespan=lifespan,
)

# Register all tools
register_tools(mcp, rag_engine)


# Resource for index stats
@mcp.resource("rag://index/stats")
async def get_index_stats() -> dict:
    """Get statistics about the RAG index."""
    return await rag_engine.get_stats()


# Resource for health status
@mcp.resource("rag://health")
async def get_health() -> dict:
    """Get health status of RAG components."""
    return await rag_engine.check_health()


# Resource for document list
@mcp.resource("rag://documents")
async def get_documents() -> dict:
    """Get list of indexed documents."""
    return await rag_engine.list_documents()


def run_server():
    """Run the MCP server with streamable HTTP transport."""
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run_server()
