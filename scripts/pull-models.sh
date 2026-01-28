#!/bin/bash
# Pull required Ollama models for the application.

set -e

echo "Pulling required Ollama models..."

echo "1/2: Pulling LLM model..."
ollama pull qwen2.5-coder:1.5b-base

echo "2/2: Pulling embedding model..."
ollama pull nomic-embed-text

echo "All models pulled successfully!"
