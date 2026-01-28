"""Entry point for `python -m local_chat_agent`."""

import uvicorn

from .config import settings


def main():
    uvicorn.run(
        "local_chat_agent.app:app",
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
