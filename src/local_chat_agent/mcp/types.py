"""MCP type definitions."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mcp import ClientSession


class TransportType(Enum):
    """Supported MCP transport types."""
    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


@dataclass
class SSEServerParameters:
    """Parameters for connecting to an SSE-based MCP server (legacy)."""
    url: str
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class StreamableHTTPServerParameters:
    """Parameters for connecting to a Streamable HTTP MCP server."""
    url: str


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
