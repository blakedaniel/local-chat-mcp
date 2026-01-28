# RAG MCP Server

A standalone RAG (Retrieval-Augmented Generation) server exposed via the Model Context Protocol (MCP). Powered by LlamaIndex, Ollama, and Qdrant.

## Features

- Ingest documents (PDF, TXT, MD, DOCX, HTML, CSV) into a vector database
- Query documents using natural language questions
- Get answers with source attribution
- Manage indexed documents (list, delete, rebuild)

## Prerequisites

- [Ollama](https://ollama.ai/) running locally with an embedding model (`nomic-embed-text`)
- [Qdrant](https://qdrant.tech/) vector database running (default: `localhost:6333`)

## Standalone Usage

```bash
# Install
pip install -e .

# Run
python -m rag_mcp_server

# Or use the entry point
rag-mcp-server
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_HOST` | `0.0.0.0` | Server bind host |
| `RAG_PORT` | `8001` | Server bind port |
| `RAG_OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `RAG_LLM_MODEL` | `qwen2.5-coder:1.5b-base` | LLM model for queries |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `RAG_QDRANT_HOST` | `localhost` | Qdrant host |
| `RAG_QDRANT_PORT` | `6333` | Qdrant port |
| `RAG_COLLECTION_NAME` | `rag_documents` | Qdrant collection |
| `RAG_DATA_DIR` | `./data` | Document directory |

## MCP Tools

- `query` - Ask questions about indexed documents
- `ingest_document` - Add a document by content
- `ingest_from_path` - Index from file/directory
- `rebuild_index` - Rebuild from data directory
- `list_documents` - List indexed documents
- `delete_document` - Remove a document
- `get_rag_stats` - Index statistics
- `check_rag_health` - Component health check
