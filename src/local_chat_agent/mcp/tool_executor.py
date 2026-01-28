"""Tool call parsing and execution for LLM-driven tool use."""

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client_manager import MCPClientManager


class ToolCallParser:
    """Parses and extracts tool calls from LLM output."""

    TOOL_CALL_PATTERN = re.compile(
        r'### TOOL_CALL:\s*(\S+)\s*\n```(?:json)?\s*\n(.*?)\n```',
        re.DOTALL,
    )

    @classmethod
    def extract_tool_calls(cls, text: str) -> list[tuple[str, dict]]:
        """Extract all tool calls from text.

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
        return tool_calls

    @classmethod
    def replace_tool_call_with_result(
        cls,
        text: str,
        tool_name: str,
        arguments: dict,
        result: str,
    ) -> str:
        """Replace a tool call in text with its result."""
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


def format_mcp_result(result) -> str:
    """Extract text from MCP result object."""
    if hasattr(result, 'content'):
        return "\n".join(
            item.text if hasattr(item, 'text') else str(item)
            for item in result.content
        )
    return str(result)


def format_tool_results_for_llm(tool_results: list[dict]) -> str:
    """Format tool results for LLM consumption."""
    sections = []
    for tr in tool_results:
        section = f"### TOOL_RESULT: {tr['tool']}\n"
        section += f"Arguments: {json.dumps(tr['arguments'])}\n"
        section += f"Status: {tr['status']}\n"
        if tr['status'] == 'success':
            section += f"```\n{tr['result']}\n```"
        else:
            section += f"Error: {tr['error']}"
        sections.append(section)
    return "\n\n".join(sections)


async def execute_tool_calls(
    mcp_manager: "MCPClientManager",
    text: str,
    max_iterations: int = 5,
) -> str:
    """Execute all tool calls found in text and return updated text with results.

    Iteratively processes tool calls until no more are found or max_iterations is reached.
    """
    for iteration in range(max_iterations):
        tool_calls = ToolCallParser.extract_tool_calls(text)
        if not tool_calls:
            break

        print(f"Executing {len(tool_calls)} tool call(s) (iteration {iteration + 1})")

        for tool_name, arguments in tool_calls:
            try:
                result = await mcp_manager.call_tool_by_name(tool_name, arguments)
                result_text = format_mcp_result(result)
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


async def agentic_execute(
    mcp_manager: "MCPClientManager",
    ollama_client,  # httpx.AsyncClient or local_chat_agent.llm.OllamaClient
    ollama_url: str,
    model_name: str,
    system_prompt: str,
    initial_prompt: str,
    max_turns: int = 5,
    temperature: float = 0.2,
    num_ctx: int = 16384,
) -> str:
    """Execute an agentic loop where the LLM can use tools and see their results.

    Returns the final LLM output after all tool calls are resolved.
    """
    tools = mcp_manager.get_all_tools()
    print(f"[Agentic] Starting loop with {len(tools)} tools: {[t.name for t in tools]}")

    conversation = initial_prompt

    for turn in range(max_turns):
        print(f"[Agentic Turn {turn + 1}/{max_turns}] Calling LLM...")

        response = await ollama_client.post(
            ollama_url,
            json={
                "model": model_name,
                "prompt": conversation,
                "system": system_prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": num_ctx},
            },
            timeout=None,
        )
        response.raise_for_status()
        llm_output = response.json().get("response", "")

        preview = llm_output[:300].replace('\n', '\\n')
        print(f"[Agentic Turn {turn + 1}] Output preview: {preview}...")

        tool_calls = ToolCallParser.extract_tool_calls(llm_output)
        if not tool_calls:
            print(f"[Agentic Turn {turn + 1}] No tool calls detected - returning final output")
            return llm_output

        print(f"[Agentic Turn {turn + 1}] Executing {len(tool_calls)} tool(s)")

        tool_results = []
        for tool_name, arguments in tool_calls:
            try:
                result = await mcp_manager.call_tool_by_name(tool_name, arguments)
                result_text = format_mcp_result(result)
                tool_results.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "status": "success",
                    "result": result_text,
                })
                print(f"  Tool '{tool_name}' succeeded")
            except Exception as e:
                tool_results.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "status": "error",
                    "error": str(e),
                })
                print(f"  Tool '{tool_name}' failed: {e}")

        results_section = format_tool_results_for_llm(tool_results)

        conversation = (
            f"{conversation}\n\n"
            f"Assistant:\n{llm_output}\n\n"
            f"System (Tool Results):\n{results_section}\n\n"
            f"Continue with your response, using the tool results above. "
            f"If you need more information, make another tool call. "
            f"Otherwise, provide your final output."
        )

    print(f"[Agentic] Max turns ({max_turns}) reached")
    return llm_output
