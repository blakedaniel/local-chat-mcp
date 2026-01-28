"""MCP Client Manager for integrating Model Context Protocol capabilities.

This module provides a client manager that connects to MCP servers and exposes
their tools for use by the LLM during code processing.

Supports stdio, SSE, and Streamable HTTP transports.
"""

import asyncio
from typing import Any, Union
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

from .types import (
    MCPTool,
    MCPServerConnection,
    SSEServerParameters,
    StreamableHTTPServerParameters,
    TransportType,
)


class MCPClientManager:
    """
    Manages MCP client connections to multiple servers.

    Handles connecting to MCP servers via stdio, SSE, or Streamable HTTP transport,
    discovering and caching available tools, executing tool calls, and formatting
    tools for LLM consumption.
    """

    def __init__(self):
        self.connections: dict[str, MCPServerConnection] = {}
        self._connection_tasks: dict[str, asyncio.Task] = {}

    @asynccontextmanager
    async def _create_stdio_connection(self, server_name: str, params: StdioServerParameters):
        """Create and yield a stdio-based MCP server connection."""
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = self._discover_tools(server_name, await session.list_tools())
                connection = MCPServerConnection(
                    session=session,
                    read_stream=read_stream,
                    write_stream=write_stream,
                    transport_type=TransportType.STDIO,
                    tools=tools,
                )
                yield connection

    @asynccontextmanager
    async def _create_sse_connection(self, server_name: str, params: SSEServerParameters):
        """Create and yield an SSE-based MCP server connection."""
        async with sse_client(params.url, params.headers) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = self._discover_tools(server_name, await session.list_tools())
                connection = MCPServerConnection(
                    session=session,
                    read_stream=read_stream,
                    write_stream=write_stream,
                    transport_type=TransportType.SSE,
                    tools=tools,
                )
                yield connection

    @asynccontextmanager
    async def _create_streamable_http_connection(self, server_name: str, params: StreamableHTTPServerParameters):
        """Create and yield a Streamable HTTP-based MCP server connection."""
        async with streamable_http_client(params.url) as (read_stream, write_stream, get_session_id):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = self._discover_tools(server_name, await session.list_tools())
                connection = MCPServerConnection(
                    session=session,
                    read_stream=read_stream,
                    write_stream=write_stream,
                    transport_type=TransportType.STREAMABLE_HTTP,
                    tools=tools,
                )
                yield connection

    def _discover_tools(self, server_name: str, tools_result) -> list[MCPTool]:
        """Extract MCPTool objects from a list_tools result."""
        return [
            MCPTool(
                server_name=server_name,
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
            )
            for tool in tools_result.tools
        ]

    @asynccontextmanager
    async def _create_connection(
        self,
        server_name: str,
        params: Union[StdioServerParameters, SSEServerParameters, StreamableHTTPServerParameters],
    ):
        """Create and yield an MCP server connection based on params type."""
        if isinstance(params, StreamableHTTPServerParameters):
            async with self._create_streamable_http_connection(server_name, params) as connection:
                yield connection
        elif isinstance(params, SSEServerParameters):
            async with self._create_sse_connection(server_name, params) as connection:
                yield connection
        else:
            async with self._create_stdio_connection(server_name, params) as connection:
                yield connection

    async def connect(
        self,
        server_name: str,
        params: Union[StdioServerParameters, SSEServerParameters, StreamableHTTPServerParameters],
    ) -> list[MCPTool]:
        """Connect to an MCP server and discover its tools.

        Args:
            server_name: A unique identifier for this server connection
            params: Connection parameters for the server

        Returns:
            List of discovered MCPTool objects
        """
        async with self._create_connection(server_name, params) as connection:
            self.connections[server_name] = connection
            print(f"Connected to MCP server '{server_name}' with {len(connection.tools)} tools")
            return connection.tools

    async def connect_persistent(
        self,
        server_name: str,
        params: Union[StdioServerParameters, SSEServerParameters, StreamableHTTPServerParameters],
    ):
        """Establish a persistent connection to an MCP server.

        This method starts the connection in a way that keeps it alive
        for the duration of the application lifecycle.
        """
        if isinstance(params, StreamableHTTPServerParameters):
            transport_type = "Streamable HTTP"
        elif isinstance(params, SSEServerParameters):
            transport_type = "SSE"
        else:
            transport_type = "stdio"
        try:
            async with self._create_connection(server_name, params) as connection:
                self.connections[server_name] = connection
                print(f"Connected to MCP server '{server_name}' ({transport_type}) with {len(connection.tools)} tools")

                # Keep connection alive until cancelled
                while True:
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            print(f"Disconnecting from MCP server '{server_name}'")
            if server_name in self.connections:
                del self.connections[server_name]
            raise
        except Exception as e:
            print(f"Error connecting to MCP server '{server_name}': {e}")
            raise

    def start_server(
        self,
        server_name: str,
        params: Union[StdioServerParameters, SSEServerParameters, StreamableHTTPServerParameters],
    ) -> asyncio.Task:
        """Start an MCP server connection as a background task.

        Returns the task so it can be cancelled during shutdown.
        """
        task = asyncio.create_task(self.connect_persistent(server_name, params))
        self._connection_tasks[server_name] = task
        return task

    async def disconnect(self, server_name: str):
        """Disconnect from a specific MCP server."""
        if server_name in self._connection_tasks:
            self._connection_tasks[server_name].cancel()
            try:
                await self._connection_tasks[server_name]
            except asyncio.CancelledError:
                pass
            del self._connection_tasks[server_name]

        if server_name in self.connections:
            del self.connections[server_name]

    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for server_name in list(self._connection_tasks.keys()):
            await self.disconnect(server_name)

    def is_connected(self, server_name: str) -> bool:
        """Check if connected to a specific server."""
        return server_name in self.connections

    def get_connected_servers(self) -> list[str]:
        """Get list of connected server names."""
        return list(self.connections.keys())

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any:
        """Call a tool on an MCP server.

        Args:
            server_name: The server to call the tool on
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments to pass to the tool

        Returns:
            The tool execution result

        Raises:
            ValueError: If not connected to the specified server
        """
        connection = self.connections.get(server_name)
        if not connection:
            raise ValueError(f"Not connected to MCP server: {server_name}")

        result = await connection.session.call_tool(tool_name, arguments)
        return result

    async def call_tool_by_name(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool by name, automatically finding the right server.

        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments to pass to the tool

        Returns:
            The tool execution result

        Raises:
            ValueError: If tool is not found on any connected server
        """
        for server_name, connection in self.connections.items():
            for tool in connection.tools:
                if tool.name == tool_name:
                    return await self.call_tool(server_name, tool_name, arguments)

        raise ValueError(f"Tool '{tool_name}' not found on any connected server")

    def get_all_tools(self) -> list[MCPTool]:
        """Get all available tools from all connected servers."""
        all_tools = []
        for connection in self.connections.values():
            all_tools.extend(connection.tools)
        return all_tools

    def get_tools_for_server(self, server_name: str) -> list[MCPTool]:
        """Get tools for a specific server."""
        connection = self.connections.get(server_name)
        if not connection:
            return []
        return connection.tools

    def format_tools_for_prompt(self) -> str:
        """Format all available tools as a string for inclusion in LLM prompts."""
        tools = self.get_all_tools()
        if not tools:
            return "No tools available."

        lines = ["Available tools:"]
        for tool in tools:
            lines.append(f"\n### {tool.name}")
            lines.append(f"Server: {tool.server_name}")
            lines.append(f"Description: {tool.description}")
            if tool.input_schema:
                props = tool.input_schema.get("properties", {})
                required = tool.input_schema.get("required", [])
                if props:
                    lines.append("Parameters:")
                    for param_name, param_info in props.items():
                        req_marker = " (required)" if param_name in required else ""
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "")
                        lines.append(f"  - {param_name}: {param_type}{req_marker}")
                        if param_desc:
                            lines.append(f"    {param_desc}")

        return "\n".join(lines)

    def format_tool_call_instructions(self) -> str:
        """Generate instructions for the LLM on how to call tools."""
        return """
To use a tool, output a tool call block in this exact format:

### TOOL_CALL: <tool_name>
```json
{
  "param1": "value1",
  "param2": "value2"
}
```

The tool will be executed and the result will be provided back to you.
You can make multiple tool calls in sequence if needed.
"""
