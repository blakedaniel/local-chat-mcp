"""Load MCP server configuration from Claude-format mcp-servers.json."""

import json
import os
import re
from typing import Union

from mcp import StdioServerParameters

from .types import SSEServerParameters, StreamableHTTPServerParameters


def _interpolate_env_vars(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))
    return re.sub(r'\$\{(\w+)\}', replacer, value)


def _interpolate_dict(d: dict) -> dict:
    """Recursively interpolate env vars in a dictionary."""
    result = {}
    for k, v in d.items():
        if isinstance(v, str):
            result[k] = _interpolate_env_vars(v)
        elif isinstance(v, dict):
            result[k] = _interpolate_dict(v)
        elif isinstance(v, list):
            result[k] = [_interpolate_env_vars(i) if isinstance(i, str) else i for i in v]
        else:
            result[k] = v
    return result


def load_mcp_config(
    config_path: str,
) -> dict[str, Union[StdioServerParameters, SSEServerParameters, StreamableHTTPServerParameters]]:
    """Load MCP server configurations from a Claude-format mcp-servers.json file.

    Supports:
    - stdio servers: {"command": "...", "args": [...], "env": {...}}
    - SSE servers: {"url": "...", "transport": "sse"}
    - Streamable HTTP servers: {"url": "...", "transport": "streamable-http"} or just {"url": "..."}

    Environment variables in the format ${VAR_NAME} are interpolated from the process environment.

    Args:
        config_path: Path to the mcp-servers.json file

    Returns:
        Dictionary mapping server names to their connection parameters
    """
    if not os.path.exists(config_path):
        print(f"MCP config not found at {config_path}, no servers configured")
        return {}

    with open(config_path, "r") as f:
        raw = json.load(f)

    servers_raw = raw.get("mcpServers", {})
    servers: dict[str, Union[StdioServerParameters, SSEServerParameters, StreamableHTTPServerParameters]] = {}

    for name, cfg in servers_raw.items():
        cfg = _interpolate_dict(cfg)

        if "command" in cfg:
            # stdio server
            env = {**os.environ, **cfg.get("env", {})}
            servers[name] = StdioServerParameters(
                command=cfg["command"],
                args=cfg.get("args", []),
                env=env,
            )
        elif "url" in cfg:
            transport = cfg.get("transport", "streamable-http")
            if transport == "sse":
                servers[name] = SSEServerParameters(
                    url=cfg["url"],
                    headers=cfg.get("headers", {}),
                )
            else:
                # Default to streamable-http
                servers[name] = StreamableHTTPServerParameters(url=cfg["url"])
        else:
            print(f"Warning: Unknown MCP server config format for '{name}', skipping")

    return servers
