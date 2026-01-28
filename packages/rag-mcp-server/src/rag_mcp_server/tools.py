"""MCP Tool definitions for the RAG server."""

from typing import Optional

from mcp.server.fastmcp import FastMCP

from .rag_engine import RAGEngine


def register_tools(mcp: FastMCP, rag_engine: RAGEngine) -> None:
    """Register all RAG tools with the MCP server."""

    @mcp.tool()
    async def query(question: str, top_k: Optional[int] = 3) -> dict:
        """Query the RAG system with a natural language question.

        Ask questions about the indexed documents and get answers with source references.

        Args:
            question: The natural language question to ask
            top_k: Number of similar document chunks to retrieve (default: 3)

        Returns:
            A dict with 'answer' (the response) and 'sources' (list of source documents)
        """
        return await rag_engine.query(question, top_k)

    @mcp.tool()
    async def ingest_document(content: str, filename: str) -> dict:
        """Ingest a document into the RAG system.

        The document will be chunked, embedded, and stored in the vector database.

        Args:
            content: The text content of the document
            filename: A name for the document (used for source attribution)

        Returns:
            A dict with status and document_id
        """
        return await rag_engine.ingest_document(content, filename)

    @mcp.tool()
    async def ingest_from_path(path: str) -> dict:
        """Ingest documents from a file or directory path on the server.

        Supports PDF, TXT, MD, DOCX, HTML, and CSV files.

        Args:
            path: Path to a file or directory containing documents

        Returns:
            A dict with status, documents_ingested count, and list of files
        """
        return await rag_engine.ingest_from_path(path)

    @mcp.tool()
    async def rebuild_index() -> dict:
        """Rebuild the entire index from the data directory.

        This will delete the existing index and re-index all documents
        in the configured data directory.

        Returns:
            A dict with status and documents_indexed count
        """
        return await rag_engine.rebuild_index()

    @mcp.tool()
    async def list_documents() -> dict:
        """List all documents currently indexed in the RAG system.

        Returns:
            A dict with 'documents' (list of document info) and 'total' count
        """
        return await rag_engine.list_documents()

    @mcp.tool()
    async def delete_document(filename: str) -> dict:
        """Delete a document from the index by filename.

        Args:
            filename: The filename of the document to delete

        Returns:
            A dict with status and message
        """
        return await rag_engine.delete_document(filename)

    @mcp.tool()
    async def get_rag_stats() -> dict:
        """Get statistics about the RAG index.

        Returns information about the current index including document count,
        chunk count, and model configuration.

        Returns:
            A dict with index statistics
        """
        return await rag_engine.get_stats()

    @mcp.tool()
    async def check_rag_health() -> dict:
        """Check the health of all RAG dependencies.

        Verifies connectivity to Qdrant, Ollama, and checks index status.

        Returns:
            A dict with health status for each component
        """
        return await rag_engine.check_health()
