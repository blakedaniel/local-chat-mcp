import os
import shutil
import zipfile
import tempfile
import asyncio
import httpx
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from mcp import StdioServerParameters

from mcp_client import MCPClientManager, execute_tool_calls

# MCP Client Manager - Global instance
mcp_manager = MCPClientManager()

# MCP Server configurations
# Add your MCP servers here. Examples:
# - Filesystem server: gives LLM access to read/write files
# - Git server: gives LLM access to git operations
# - Custom servers: any MCP-compatible server
MCP_SERVERS: dict[str, StdioServerParameters] = {
    # Example configurations (uncomment and modify as needed):
    #
    # "filesystem": StdioServerParameters(
    #     command="npx",
    #     args=["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    # ),
    #
    # "git": StdioServerParameters(
    #     command="npx",
    #     args=["-y", "@modelcontextprotocol/server-git"]
    # ),
    #
    # "fetch": StdioServerParameters(
    #     command="npx",
    #     args=["-y", "@modelcontextprotocol/server-fetch"]
    # ),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles MCP server connections."""
    # Startup: Connect to configured MCP servers
    connection_tasks = []
    for server_name, params in MCP_SERVERS.items():
        try:
            task = mcp_manager.start_server(server_name, params)
            connection_tasks.append(task)
            # Give the server a moment to initialize
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Warning: Failed to start MCP server '{server_name}': {e}")

    if MCP_SERVERS:
        print(f"MCP Client initialized with {len(mcp_manager.get_connected_servers())} server(s)")
        tools = mcp_manager.get_all_tools()
        if tools:
            print(f"Available tools: {[t.name for t in tools]}")
    else:
        print("No MCP servers configured. Add servers to MCP_SERVERS dict to enable tools.")

    yield

    # Shutdown: Disconnect from all MCP servers
    await mcp_manager.disconnect_all()
    print("MCP connections closed")


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# Configuration
OLLAMA_URL = "http://ollama:11434/api/generate"
# 14B is crucial here. 3B struggles to generate multi-file projects consistently.
MODEL_NAME = "qwen2.5-coder:14b" 

ALLOWED_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', 
    '.java', '.cpp', '.c', '.h', '.rs', '.go', '.php', '.rb',
    '.json', '.yaml', '.yml', '.sql', '.md', '.txt'
}

def parse_and_save_files(raw_text: str, base_dir: str):
    """
    Scans for '### FILE: <name>' markers. 
    If found, saves multiple files. Returns list of created files.
    """
    # Regex to split by the file delimiter
    # It captures the filename in group 1
    segments = re.split(r'### FILE:\s*([^\n]+)\n', raw_text)

    # If we didn't find at least one split (3 segments: preamble, filename, content), 
    # then it's a standard single-file response.
    if len(segments) < 3:
        return None 

    created_files = []
    
    # Segments structure: [preamble, filename1, content1, filename2, content2...]
    # We skip index 0 (preamble) and iterate by 2
    for i in range(1, len(segments), 2):
        fname = segments[i].strip()
        content = segments[i+1]
        
        # Cleanup markdown code blocks from the content if present
        if content.strip().startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].startswith("```"): lines = lines[:-1]
            content = "\n".join(lines)

        # Security: Prevent writing to parent directories
        if ".." in fname or fname.startswith("/") or fname.startswith("\\"):
            print(f"⚠️ Skipping unsafe filename: {fname}")
            continue

        full_path = os.path.join(base_dir, fname)
        
        # Create subdirectories if needed (e.g., src/main/java/...)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        created_files.append(fname)
        
    return created_files

async def process_file(
    file_path: str,
    user_instructions: str,
    client: httpx.AsyncClient,
    extract_root: str,
    use_mcp_tools: bool = True
):
    """
    Process a single file through the LLM with optional MCP tool support.

    Args:
        file_path: Path to the file to process
        user_instructions: User's refactoring instructions
        client: httpx async client for Ollama requests
        extract_root: Root directory for extracted files
        use_mcp_tools: Whether to include MCP tools in the prompt
    """
    filename = os.path.basename(file_path)
    if os.path.splitext(filename)[1].lower() not in ALLOWED_EXTENSIONS:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return

    # --- BUILD SYSTEM PROMPT WITH MCP TOOLS ---
    base_system_prompt = (
        "You are an expert software architect. "
        "If the user asks to PORT or REWRITE code into a new language/framework that requires multiple files (like Spring Boot), "
        "you MUST output every single file needed for the new project.\n\n"
        "STRICT OUTPUT FORMAT:\n"
        "To create a file, start with: ### FILE: <path/to/filename>\n"
        "Followed immediately by the code for that file.\n"
        "Example:\n"
        "### FILE: pom.xml\n"
        "<project>...</project>\n"
        "### FILE: src/main/java/com/example/App.java\n"
        "package com.example;\n"
        "...\n\n"
        "Do not output conversational text. Just the file markers and code."
    )

    # Add MCP tool information if tools are available
    mcp_tools_section = ""
    if use_mcp_tools and mcp_manager.get_all_tools():
        mcp_tools_section = (
            "\n\n--- AVAILABLE TOOLS ---\n"
            "You have access to the following tools that you can use to gather information "
            "or perform actions before generating code:\n\n"
            f"{mcp_manager.format_tools_for_prompt()}\n\n"
            f"{mcp_manager.format_tool_call_instructions()}\n"
            "Use tools when you need external information. After tool results are provided, "
            "continue with your code generation.\n"
            "--- END TOOLS ---\n"
        )

    system_prompt = base_system_prompt + mcp_tools_section
    
    user_prompt = (
        f"CURRENT FILE: {filename}\n"
        f"INSTRUCTION: {user_instructions}\n"
        f"CONTENT:\n{content}"
    )

    try:
        response = await client.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 16384
                }
            },
            timeout=None  # <--- DISABLES timeout completely
        )
        response.raise_for_status()
        result = response.json()
        raw_output = result.get("response", "")

        # --- EXECUTE MCP TOOL CALLS IF PRESENT ---
        if use_mcp_tools and mcp_manager.get_all_tools():
            raw_output = await execute_tool_calls(mcp_manager, raw_output)

        # --- PARSING LOGIC ---

        # 1. Try to parse as Multi-File output
        new_files = parse_and_save_files(raw_output, extract_root)

        if new_files:
            print(f"✅ Converted {filename} into {len(new_files)} new files:")
            for nf in new_files:
                print(f"   -> {nf}")

            # OPTIONAL: Delete the original file if it was successfully ported
            # os.remove(file_path)
            # print(f"   (Removed original {filename})")

        else:
            # 2. Fallback: Treat as Single-File Refactor (Standard 1-to-1)
            # (Use previous cleaning logic for single files)
            cleaned_code = raw_output.strip()
            if cleaned_code.startswith("```"):
                cleaned_code = cleaned_code.split("\n", 1)[1].rsplit("\n", 1)[0]

            if len(cleaned_code) > 0 and "I cannot assist" not in cleaned_code:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_code)
                print(f"✅ Refactored {filename} (Single file update)")

    except Exception as e:
        import traceback
        print(f"❌ Error processing {filename}: {type(e).__name__} - {e}")
        traceback.print_exc()  # Prints the full stack trace to logs

# --- MCP Management Endpoints ---

@app.get("/mcp/servers")
async def list_mcp_servers():
    """List all connected MCP servers and their tools."""
    servers = {}
    for server_name in mcp_manager.get_connected_servers():
        tools = mcp_manager.get_tools_for_server(server_name)
        servers[server_name] = {
            "connected": True,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema
                }
                for tool in tools
            ]
        }
    return {"servers": servers, "total_tools": len(mcp_manager.get_all_tools())}


@app.get("/mcp/tools")
async def list_mcp_tools():
    """List all available MCP tools across all servers."""
    tools = mcp_manager.get_all_tools()
    return {
        "tools": [
            {
                "server": tool.server_name,
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            }
            for tool in tools
        ]
    }


@app.post("/mcp/servers/{server_name}/connect")
async def connect_mcp_server(server_name: str, command: str = Form(...), args: str = Form("")):
    """
    Connect to a new MCP server.

    Args:
        server_name: Unique identifier for the server
        command: Command to launch the server (e.g., "npx")
        args: Space-separated arguments (e.g., "-y @modelcontextprotocol/server-filesystem /workspace")
    """
    if mcp_manager.is_connected(server_name):
        return JSONResponse(
            status_code=400,
            content={"error": f"Server '{server_name}' is already connected"}
        )

    try:
        args_list = args.split() if args else []
        params = StdioServerParameters(command=command, args=args_list)

        mcp_manager.start_server(server_name, params)
        # Give it a moment to connect
        await asyncio.sleep(1)

        if mcp_manager.is_connected(server_name):
            tools = mcp_manager.get_tools_for_server(server_name)
            return {
                "status": "connected",
                "server": server_name,
                "tools": [t.name for t in tools]
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Connection started but server not ready"}
            )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to connect: {str(e)}"}
        )


@app.post("/mcp/servers/{server_name}/disconnect")
async def disconnect_mcp_server(server_name: str):
    """Disconnect from an MCP server."""
    if not mcp_manager.is_connected(server_name):
        return JSONResponse(
            status_code=404,
            content={"error": f"Server '{server_name}' is not connected"}
        )

    await mcp_manager.disconnect(server_name)
    return {"status": "disconnected", "server": server_name}


@app.post("/mcp/tools/{tool_name}/call")
async def call_mcp_tool(tool_name: str, arguments: dict = {}):
    """
    Directly call an MCP tool.

    Args:
        tool_name: Name of the tool to call
        arguments: JSON body with tool arguments
    """
    try:
        result = await mcp_manager.call_tool_by_name(tool_name, arguments)

        # Extract content from result
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


# --- Standard Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/refactor")
async def refactor_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    instructions: str = Form(...)
):
    work_dir = tempfile.mkdtemp()
    upload_zip = os.path.join(work_dir, "input.zip")
    extract_dir = os.path.join(work_dir, "source")
    output_zip = os.path.join(work_dir, "refactored.zip")
    
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with open(upload_zip, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        with zipfile.ZipFile(upload_zip, 'r') as z:
            z.extractall(extract_dir)

        files_to_process = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                files_to_process.append(os.path.join(root, file))

        sem = asyncio.Semaphore(1)

        async def worker(fp, instr, client):
            async with sem:
                # UPDATED: Passing extract_dir so it knows where to build the new project tree
                await process_file(fp, instr, client, extract_dir)

        async with httpx.AsyncClient() as client:
            tasks = [worker(fp, instructions, client) for fp in files_to_process]
            await asyncio.gather(*tasks)

        # Re-zip the ENTIRE extracted folder (which now contains new folders like src/main/java)
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, extract_dir)
                    z.write(file_path, arcname)

        return FileResponse(output_zip, filename="converted_project.zip", media_type="application/zip")

    except Exception as e:
        return {"error": str(e)}
        
    finally:
        background_tasks.add_task(shutil.rmtree, work_dir)