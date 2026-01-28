FROM python:3.11-slim

# Install Node.js 20 (required for npx-based MCP servers like GitHub)
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
COPY packages/rag-mcp-server/pyproject.toml packages/rag-mcp-server/
COPY packages/rag-mcp-server/src/ packages/rag-mcp-server/src/
COPY src/ src/

RUN pip install --no-cache-dir .

# Copy remaining files
COPY templates/ templates/
COPY mcp-servers.json .
COPY data/ data/

EXPOSE 8000

CMD ["python", "-m", "local_chat_agent"]
