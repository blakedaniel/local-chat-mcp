"""Application configuration."""

import os
from dataclasses import dataclass
from pathlib import Path

import dotenv

dotenv.load_dotenv()


# Project root is two levels up from this file (src/local_chat_agent/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class AppSettings:
    """Main application settings with environment variable overrides."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Ollama
    ollama_url: str = "http://ollama:11434/api/generate"
    model_name: str = "qwen2.5-coder:1.5b-base"

    # Paths
    templates_dir: str = str(PROJECT_ROOT / "templates")
    mcp_config_path: str = str(PROJECT_ROOT / "mcp-servers.json")

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            host=os.getenv("APP_HOST", cls.host),
            port=int(os.getenv("APP_PORT", cls.port)),
            ollama_url=os.getenv("OLLAMA_URL", cls.ollama_url),
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            templates_dir=os.getenv("TEMPLATES_DIR", cls.templates_dir),
            mcp_config_path=os.getenv("MCP_CONFIG_PATH", cls.mcp_config_path),
        )


settings = AppSettings.from_env()
