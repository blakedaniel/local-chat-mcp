# Local Chat Agent

LLM-powered code refactoring tool with MCP (Model Context Protocol) integration. Features RAG-augmented code generation using local LLMs via Ollama.

## Architecture

```
local-chat-agent/
├── packages/rag-mcp-server/    # Standalone RAG MCP Server (publishable)
├── src/local_chat_agent/       # Main FastAPI application
│   ├── llm/                    # LLM abstraction (Ollama backend)
│   ├── mcp/                    # MCP client manager + config loader
│   └── routes/                 # Chat & MCP management endpoints
├── templates/                  # Web UI
├── scripts/                    # Dev/ops scripts
└── data/                       # RAG documents
```

## Quick Start

### Docker (recommended)

```bash
docker compose up --build -d

# Pull required models
docker exec -it ollama_backend ollama pull qwen2.5-coder:1.5b-base
docker exec -it ollama_backend ollama pull nomic-embed-text
```

Visit `http://localhost:8000`.

### GPU Support

```bash
docker compose -f docker-compose.gpu.yml up --build -d
```

### Local Development

```bash
# Prerequisites: Ollama running locally, Docker (for Qdrant)
./scripts/start-dev.sh
```

## MCP Configuration

MCP servers are configured in `mcp-servers.json` (Claude-compatible format):

```json
{
  "mcpServers": {
    "rag-server": {
      "url": "http://localhost:8001/mcp",
      "transport": "streamable-http"
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

Environment variables use `${VAR}` syntax and are interpolated from the process environment.

## RAG MCP Server

The RAG server is an independent package in `packages/rag-mcp-server/`. See its [README](packages/rag-mcp-server/README.md) for standalone usage.

## Configuration

Copy `.env.example` to `.env` and adjust as needed. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://ollama:11434/api/generate` | Ollama API endpoint |
| `MODEL_NAME` | `qwen2.5-coder:1.5b-base` | LLM model for refactoring |
| `MCP_CONFIG_PATH` | `./mcp-servers.json` | Path to MCP server config |

See `.env.example` for all available settings.
