"""FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx

# Configure MCP monitoring logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("mcp.monitor").setLevel(logging.INFO)
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .config import settings
from .mcp.client_manager import MCPClientManager
from .mcp.config_loader import load_mcp_config
from .routes import chat, mcp as mcp_routes

# Global MCP Client Manager
mcp_manager = MCPClientManager()


async def _check_ollama_connectivity():
    """Attempt a lightweight health check against the configured Ollama instance."""
    # Derive base URL from the generate endpoint (strip /api/generate)
    base_url = settings.ollama_url.rsplit("/api/", 1)[0]
    print(f"[startup] Ollama URL: {settings.ollama_url}")
    print(f"[startup] Model: {settings.model_name}")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(base_url, timeout=5)
            if resp.status_code == 200:
                print(f"[startup] Ollama is reachable at {base_url}")
            else:
                print(f"[startup] WARNING: Ollama returned status {resp.status_code} at {base_url}")
    except Exception as e:
        print(f"[startup] WARNING: Cannot reach Ollama at {base_url}: {e}")
        print("[startup] Refactor requests will fail until Ollama is available.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles MCP server connections."""
    # Check Ollama connectivity
    await _check_ollama_connectivity()

    # Load MCP config and start servers
    mcp_servers = load_mcp_config(settings.mcp_config_path)

    for server_name, params in mcp_servers.items():
        try:
            mcp_manager.start_server(server_name, params)
        except Exception as e:
            print(f"Warning: Failed to start MCP server '{server_name}': {e}")

    if mcp_servers:
        # Wait for connections to establish with retries
        for attempt in range(10):
            await asyncio.sleep(1)
            connected = mcp_manager.get_connected_servers()
            if len(connected) >= len(mcp_servers):
                break
        print(f"MCP Client initialized with {len(mcp_manager.get_connected_servers())} server(s)")
        tools = mcp_manager.get_all_tools()
        if tools:
            print(f"Available tools: {[t.name for t in tools]}")
    else:
        print("No MCP servers configured. Add servers to mcp-servers.json to enable tools.")

    # Wire the MCP manager into the routes module
    mcp_routes.set_mcp_manager(mcp_manager)

    yield

    # Shutdown
    await mcp_manager.disconnect_all()
    print("MCP connections closed")


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=settings.templates_dir)

# Include route modules
app.include_router(chat.router)
app.include_router(mcp_routes.router)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
