"""Ollama LLM client implementation."""

import httpx

from .base import BaseLLMClient
from .types import GenerationConfig


class OllamaClient(BaseLLMClient):
    """LLM client that talks to an Ollama instance."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: httpx.AsyncClient | None = None

    @property
    def generate_url(self) -> str:
        # Support both full URL (http://host/api/generate) and base URL (http://host:11434)
        if self.base_url.endswith("/api/generate"):
            return self.base_url
        return f"{self.base_url}/api/generate"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def generate(
        self,
        prompt: str,
        system: str = "",
        config: GenerationConfig | None = None,
    ) -> str:
        cfg = config or GenerationConfig()
        model = cfg.model or self.model

        client = await self._get_client()
        response = await client.post(
            self.generate_url,
            json={
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": cfg.stream,
                "options": {
                    "temperature": cfg.temperature,
                    "num_ctx": cfg.num_ctx,
                },
            },
            timeout=None,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
