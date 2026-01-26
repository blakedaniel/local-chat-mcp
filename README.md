# AI DevOps Agent

An autonomous agent that refactors code from one GitHub repository and pushes the results to a new repository.

## Setup
1. Run \`docker compose up --build -d\`
2. Visit \`http://localhost:8000\`
3. Download the model: \`docker exec -it ollama_backend ollama pull qwen2.5-coder:14b\`

## Usage
- **Source URL:** The repo to read.
- **New Repo Name:** The name of the NEW repo to create.
- **Token:** GitHub PAT with \`repo\` scope.
