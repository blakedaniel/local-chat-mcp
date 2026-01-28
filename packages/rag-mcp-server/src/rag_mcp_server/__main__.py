"""Entry point for `python -m rag_mcp_server`."""

from .server import run_server


def main():
    print("=" * 50)
    print("RAG MCP Server")
    print("LlamaIndex + Ollama + Qdrant")
    print("=" * 50)
    print()
    run_server()


if __name__ == "__main__":
    main()
