"""MCP management endpoints."""

import asyncio

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from mcp import StdioServerParameters

from ..mcp.types import SSEServerParameters
from ..mcp.client_manager import MCPClientManager

router = APIRouter(prefix="/mcp", tags=["mcp"])


def get_mcp_manager() -> MCPClientManager:
    """Get the MCP manager from app state. Set at app startup."""
    return _mcp_manager


_mcp_manager: MCPClientManager = None  # type: ignore


def set_mcp_manager(manager: MCPClientManager):
    global _mcp_manager
    _mcp_manager = manager


@router.get("/servers")
async def list_mcp_servers():
    """List all connected MCP servers and their tools."""
    manager = get_mcp_manager()
    servers = {}
    for server_name in manager.get_connected_servers():
        tools = manager.get_tools_for_server(server_name)
        servers[server_name] = {
            "connected": True,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in tools
            ],
        }

    return {"servers": servers, "total_tools": len(manager.get_all_tools())}


@router.get("/tools")
async def list_mcp_tools():
    """List all available MCP tools across all servers."""
    manager = get_mcp_manager()
    tools = manager.get_all_tools()
    return {
        "tools": [
            {
                "server": tool.server_name,
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]
    }


@router.post("/servers/{server_name}/connect")
async def connect_mcp_server(server_name: str, command: str = Form(...), args: str = Form("")):
    """Connect to a new stdio-based MCP server."""
    manager = get_mcp_manager()

    if manager.is_connected(server_name):
        return JSONResponse(
            status_code=400,
            content={"error": f"Server '{server_name}' is already connected"},
        )

    try:
        args_list = args.split() if args else []
        params = StdioServerParameters(command=command, args=args_list)

        manager.start_server(server_name, params)
        await asyncio.sleep(1)

        if manager.is_connected(server_name):
            tools = manager.get_tools_for_server(server_name)
            return {
                "status": "connected",
                "server": server_name,
                "transport": "stdio",
                "tools": [t.name for t in tools],
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Connection started but server not ready"},
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to connect: {str(e)}"},
        )


@router.post("/servers/{server_name}/connect-sse")
async def connect_mcp_sse_server(server_name: str, url: str = Form(...)):
    """Connect to a new SSE-based MCP server."""
    manager = get_mcp_manager()

    if manager.is_connected(server_name):
        return JSONResponse(
            status_code=400,
            content={"error": f"Server '{server_name}' is already connected"},
        )

    try:
        params = SSEServerParameters(url=url)
        manager.start_server(server_name, params)
        await asyncio.sleep(2)

        if manager.is_connected(server_name):
            tools = manager.get_tools_for_server(server_name)
            return {
                "status": "connected",
                "server": server_name,
                "transport": "sse",
                "url": url,
                "tools": [t.name for t in tools],
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Connection started but server not ready"},
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to connect: {str(e)}"},
        )


@router.post("/servers/{server_name}/disconnect")
async def disconnect_mcp_server(server_name: str):
    """Disconnect from an MCP server."""
    manager = get_mcp_manager()

    if not manager.is_connected(server_name):
        return JSONResponse(
            status_code=404,
            content={"error": f"Server '{server_name}' is not connected"},
        )

    await manager.disconnect(server_name)
    return {"status": "disconnected", "server": server_name}


@router.post("/tools/{tool_name}/call")
async def call_mcp_tool(tool_name: str, arguments: dict = {}):
    """Directly call an MCP tool."""
    manager = get_mcp_manager()

    try:
        result = await manager.call_tool_by_name(tool_name, arguments)

        if hasattr(result, 'content'):
            content = [
                {"type": "text", "text": item.text} if hasattr(item, 'text') else {"type": "unknown", "value": str(item)}
                for item in result.content
            ]
        else:
            content = [{"type": "text", "text": str(result)}]

        return {"tool": tool_name, "result": content}

    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Tool execution failed: {str(e)}"})
