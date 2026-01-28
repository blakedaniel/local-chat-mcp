"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .config import settings
from .mcp.client_manager import MCPClientManager
from .mcp.config_loader import load_mcp_config
from .routes import chat, mcp as mcp_routes

# Global MCP Client Manager
mcp_manager = MCPClientManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles MCP server connections."""
    # Load MCP config and start servers
    mcp_servers = load_mcp_config(settings.mcp_config_path)

    for server_name, params in mcp_servers.items():
        try:
            mcp_manager.start_server(server_name, params)
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Warning: Failed to start MCP server '{server_name}': {e}")

    if mcp_servers:
        # Wait a moment for connections to establish
        await asyncio.sleep(1)
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
