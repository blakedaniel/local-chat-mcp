"""
MCP Client Manager for integrating Model Context Protocol capabilities.

This module provides a client manager that connects to MCP servers and exposes
their tools for use by the LLM during code processing.

Supports both stdio and SSE (Server-Sent Events) transports.
"""

import json
import re
import asyncio
from dataclasses import dataclass, field
from typing import Any, Union
from contextlib import asynccontextmanager
from enum import Enum

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client


class TransportType(Enum):
    """Supported MCP transport types."""
    STDIO = "stdio"
    SSE = "sse"


@dataclass
class SSEServerParameters:
    """Parameters for connecting to an SSE-based MCP server."""
    url: str
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class MCPTool:
    """Represents an MCP tool with its metadata."""
    server_name: str
    name: str
    description: str
    input_schema: dict


@dataclass
class MCPServerConnection:
    """Holds connection state for an MCP server."""
    session: ClientSession
    read_stream: Any
    write_stream: Any
    transport_type: TransportType = TransportType.STDIO
    tools: list[MCPTool] = field(default_factory=list)


class MCPClientManager:
    """
    Manages MCP client connections to multiple servers.

    This class handles:
    - Connecting to MCP servers via stdio or SSE transport
    - Discovering and caching available tools
    - Executing tool calls and returning results
    - Formatting tools for LLM consumption
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

                # Discover available tools
                tools_result = await session.list_tools()
                tools = [
                    MCPTool(
                        server_name=server_name,
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                    )
                    for tool in tools_result.tools
                ]

                connection = MCPServerConnection(
                    session=session,
                    read_stream=read_stream,
                    write_stream=write_stream,
                    transport_type=TransportType.STDIO,
                    tools=tools
                )

                yield connection

    @asynccontextmanager
    async def _create_sse_connection(self, server_name: str, params: SSEServerParameters):
        """Create and yield an SSE-based MCP server connection."""
        async with sse_client(params.url, params.headers) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Discover available tools
                tools_result = await session.list_tools()
                tools = [
                    MCPTool(
                        server_name=server_name,
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                    )
                    for tool in tools_result.tools
                ]

                connection = MCPServerConnection(
                    session=session,
                    read_stream=read_stream,
                    write_stream=write_stream,
                    transport_type=TransportType.SSE,
                    tools=tools
                )

                yield connection

    @asynccontextmanager
    async def _create_connection(
        self,
        server_name: str,
        params: Union[StdioServerParameters, SSEServerParameters]
    ):
        """Create and yield an MCP server connection based on params type."""
        if isinstance(params, SSEServerParameters):
            async with self._create_sse_connection(server_name, params) as connection:
                yield connection
        else:
            async with self._create_stdio_connection(server_name, params) as connection:
                yield connection

    async def connect(
        self,
        server_name: str,
        params: Union[StdioServerParameters, SSEServerParameters]
    ) -> list[MCPTool]:
        """
        Connect to an MCP server and discover its tools.

        Args:
            server_name: A unique identifier for this server connection
            params: StdioServerParameters or SSEServerParameters for the server

        Returns:
            List of discovered MCPTool objects

        Raises:
            Exception: If connection fails
        """
        async with self._create_connection(server_name, params) as connection:
            self.connections[server_name] = connection
            print(f"Connected to MCP server '{server_name}' with {len(connection.tools)} tools")
            return connection.tools

    async def connect_persistent(
        self,
        server_name: str,
        params: Union[StdioServerParameters, SSEServerParameters]
    ):
        """
        Establish a persistent connection to an MCP server.

        This method starts the connection in a way that keeps it alive
        for the duration of the application lifecycle.
        """
        transport_type = "SSE" if isinstance(params, SSEServerParameters) else "stdio"
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
        params: Union[StdioServerParameters, SSEServerParameters]
    ) -> asyncio.Task:
        """
        Start an MCP server connection as a background task.

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
        """
        Call a tool on an MCP server.

        Args:
            server_name: The server to call the tool on
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments to pass to the tool

        Returns:
            The tool execution result

        Raises:
            ValueError: If not connected to the specified server
            Exception: If tool execution fails
        """
        connection = self.connections.get(server_name)
        if not connection:
            raise ValueError(f"Not connected to MCP server: {server_name}")

        result = await connection.session.call_tool(tool_name, arguments)
        return result

    async def call_tool_by_name(self, tool_name: str, arguments: dict) -> Any:
        """
        Call a tool by name, automatically finding the right server.

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
        """
        Format all available tools as a string for inclusion in LLM prompts.

        Returns:
            A formatted string describing all available tools
        """
        tools = self.get_all_tools()
        if not tools:
            return "No tools available."

        lines = ["Available tools:"]
        for tool in tools:
            lines.append(f"\n### {tool.name}")
            lines.append(f"Server: {tool.server_name}")
            lines.append(f"Description: {tool.description}")
            if tool.input_schema:
                # Format the input schema in a readable way
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
        """
        Generate instructions for the LLM on how to call tools.

        Returns:
            A formatted string with tool calling instructions
        """
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


class ToolCallParser:
    """Parses and extracts tool calls from LLM output."""

    # Pattern to match tool calls in LLM output
    TOOL_CALL_PATTERN = re.compile(
        r'### TOOL_CALL:\s*(\S+)\s*\n```(?:json)?\s*\n(.*?)\n```',
        re.DOTALL
    )

    @classmethod
    def extract_tool_calls(cls, text: str) -> list[tuple[str, dict]]:
        """
        Extract all tool calls from text.

        Args:
            text: The LLM output text to parse

        Returns:
            List of (tool_name, arguments) tuples
        """
        tool_calls = []

        for match in cls.TOOL_CALL_PATTERN.finditer(text):
            tool_name = match.group(1).strip()
            args_str = match.group(2).strip()

            try:
                arguments = json.loads(args_str)
                tool_calls.append((tool_name, arguments))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse tool call arguments for '{tool_name}': {e}")
                continue

        return tool_calls

    @classmethod
    def replace_tool_call_with_result(
        cls,
        text: str,
        tool_name: str,
        arguments: dict,
        result: str
    ) -> str:
        """
        Replace a tool call in text with its result.

        Args:
            text: The original text containing the tool call
            tool_name: Name of the tool that was called
            arguments: The arguments that were passed
            result: The result to insert

        Returns:
            Text with the tool call replaced by the result
        """
        # Find and replace the specific tool call
        def replacer(match):
            if match.group(1).strip() == tool_name:
                try:
                    match_args = json.loads(match.group(2).strip())
                    if match_args == arguments:
                        return f"### TOOL_RESULT: {tool_name}\n```\n{result}\n```"
                except json.JSONDecodeError:
                    pass
            return match.group(0)

        return cls.TOOL_CALL_PATTERN.sub(replacer, text)


async def execute_tool_calls(
    mcp_manager: MCPClientManager,
    text: str,
    max_iterations: int = 5
) -> str:
    """
    Execute all tool calls found in text and return updated text with results.

    This function iteratively processes tool calls until no more are found
    or max_iterations is reached.

    Args:
        mcp_manager: The MCPClientManager instance to use
        text: The text containing potential tool calls
        max_iterations: Maximum number of tool call iterations

    Returns:
        The text with tool calls replaced by their results
    """
    for iteration in range(max_iterations):
        tool_calls = ToolCallParser.extract_tool_calls(text)

        if not tool_calls:
            break

        print(f"Executing {len(tool_calls)} tool call(s) (iteration {iteration + 1})")

        for tool_name, arguments in tool_calls:
            try:
                result = await mcp_manager.call_tool_by_name(tool_name, arguments)

                # Extract text content from result
                if hasattr(result, 'content'):
                    result_text = "\n".join(
                        item.text if hasattr(item, 'text') else str(item)
                        for item in result.content
                    )
                else:
                    result_text = str(result)

                text = ToolCallParser.replace_tool_call_with_result(
                    text, tool_name, arguments, result_text
                )
                print(f"  Tool '{tool_name}' executed successfully")

            except Exception as e:
                error_msg = f"Error executing tool: {e}"
                text = ToolCallParser.replace_tool_call_with_result(
                    text, tool_name, arguments, error_msg
                )
                print(f"  Tool '{tool_name}' failed: {e}")

    return text
