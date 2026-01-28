"""Types for the LLM abstraction layer."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatMessage:
    """A message in a chat conversation."""
    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""
    temperature: float = 0.2
    num_ctx: int = 16384
    stream: bool = False
    model: Optional[str] = None  # Override default model
