#!/bin/bash
# Start the application and dependencies for local development.
# Assumes Ollama is already installed and running locally.

set -e

# Install dependencies
echo "Installing Python dependencies..."
pip install -e .
pip install -e packages/rag-mcp-server

# Start Qdrant container if not running
if ! docker ps --format '{{.Names}}' | grep -q '^qdrant_backend$'; then
    echo "Starting Qdrant vector database..."
    docker run -d \
        --name qdrant_backend \
        -p 6333:6333 \
        -p 6334:6334 \
        -v qdrant_storage:/qdrant/storage \
        qdrant/qdrant:latest
    echo "Waiting for Qdrant to be ready..."
    sleep 3
else
    echo "Qdrant already running"
fi

# Ensure nomic-embed-text model is available in Ollama
echo "Checking Ollama embedding model..."
if ! ollama list | grep -q 'nomic-embed-text'; then
    echo "Pulling nomic-embed-text model..."
    ollama pull nomic-embed-text
fi

# Create data directory if it doesn't exist
mkdir -p data

# Start RAG MCP server in background
echo "Starting RAG MCP server on port 8001..."
python -m rag_mcp_server &
RAG_PID=$!
sleep 2

# Trap to cleanup background processes on exit
cleanup() {
    echo "Shutting down..."
    kill $RAG_PID 2>/dev/null || true
    sleep 2
    if ps -p $RAG_PID > /dev/null 2>&1; then
        echo "Force killing RAG server (PID $RAG_PID)..."
        kill -9 $RAG_PID 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start the main app
echo "Starting main app on port 8000..."
python -m local_chat_agent
