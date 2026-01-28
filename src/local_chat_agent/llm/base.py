"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod

from .types import GenerationConfig


class BaseLLMClient(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str = "",
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            config: Optional generation configuration overrides.

        Returns:
            The generated text response.
        """
        ...
