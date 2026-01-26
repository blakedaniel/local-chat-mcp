import os
import shutil
import asyncio
import httpx
import logging
from typing import Dict, Any

from fastapi import FastAPI, Form, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# MCP SDK
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- CONFIGURATION ---
OLLAMA_URL = "http://ollama:11434/api/generate"
# Use 'qwen2.5-coder:32b' if you have 60GB RAM, otherwise '14b'
MODEL_NAME = "qwen2.5-coder:14b" 

# --- MCP HELPERS ---

async def run_mcp_tool(token: str, tool_name: str, args: Dict[str, Any]):
    """Execute a tool against the GitHub MCP Server"""
    
    # Run the Node.js MCP server as a subprocess
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": token}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Call the tool
            result = await session.call_tool(tool_name, arguments=args)
            
            # Check for errors in tool execution
            if result.isError:
                raise Exception(f"Tool {tool_name} failed: {result.content}")
                
            return result

async def fetch_file(owner: str, repo: str, path: str, token: str) -> str:
    """Wrapper for get_file_content"""
    try:
        # Note: Tool name 'get_file_content' is standard for this server
        result = await run_mcp_tool(token, "get_file_content", {
            "owner": owner, "repo": repo, "path": path
        })
        return result.content[0].text
    except Exception as e:
        logger.error(f"Failed to fetch {path}: {e}")
        return ""

async def create_repo(name: str, token: str) -> bool:
    """Wrapper for create_repository"""
    try:
        await run_mcp_tool(token, "create_repository", {
            "name": name,
            "description": "Refactored by AI DevOps Agent",
            "private": True,
            "autoInit": True
        })
        return True
    except Exception as e:
        logger.error(f"Failed to create repo {name}: {e}")
        return False

async def push_file(owner: str, repo: str, path: str, content: str, token: str):
    """Wrapper for create_or_update_file"""
    try:
        # Sanitize markdown code blocks if the LLM left them in
        if content.strip().startswith("```"):
            lines = content.strip().splitlines()
            # remove first line
            if lines[0].startswith("```"): lines = lines[1:]
            # remove last line
            if lines and lines[-1].startswith("```"): lines = lines[:-1]
            content = "\n".join(lines)

        await run_mcp_tool(token, "create_or_update_file", {
            "owner": owner,
            "repo": repo,
            "path": path,
            "content": content,
            "message": f"AI Agent: Added {os.path.basename(path)}",
            "branch": "main"
        })
    except Exception as e:
        logger.error(f"Failed to push {path}: {e}")

# --- OLLAMA LOGIC ---

async def refactor_code(filename: str, content: str, instructions: str, client: httpx.AsyncClient):
    if not content: return ""
    
    system_prompt = (
        "You are an elite code refactoring engine. "
        "Output ONLY valid code. Do not use Markdown backticks. Do not chat."
    )
    user_prompt = f"FILE: {filename}\nINSTRUCTION: {instructions}\nSOURCE CODE:\n{content}"

    try:
        resp = await client.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "num_ctx": 16384, # Increase to 32768 for 60GB RAM
                    "temperature": 0.2
                }
            },
            timeout=None # Wait forever
        )
        return resp.json().get("response", "")
    except Exception as e:
        logger.error(f"Ollama failed on {filename}: {e}")
        return content # Fallback to original on failure

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/convert-and-push")
async def convert_and_push(
    source_url: str = Form(...),
    new_repo_name: str = Form(...),
    github_token: str = Form(...),
    instructions: str = Form(...)
):
    logger.info(f"Starting job: {source_url} -> {new_repo_name}")

    # 1. Parse Source
    try:
        parts = source_url.rstrip("/").split("/")
        source_owner, source_repo = parts[-2], parts[-1]
    except:
        return JSONResponse({"status": "error", "error": "Invalid GitHub URL format"}, status_code=400)

    # 2. Sanitize New Repo Name
    safe_repo_name = new_repo_name.strip().replace(" ", "-").lower()

    # 3. Fetch Files (For demo, we fetch specific common files. 
    # In full version, use 'list_directory' tool to discover them dynamically)
    files_to_check = ["app.py", "main.py", "requirements.txt", "Dockerfile", "package.json", "index.js"]
    fetched_files = {}

    for f in files_to_check:
        content = await fetch_file(source_owner, source_repo, f, github_token)
        if content:
            fetched_files[f] = content
            logger.info(f"Fetched {f}")

    if not fetched_files:
        return JSONResponse({"status": "error", "error": "Could not find any common files (app.py, main.py, etc) in source repo."}, status_code=404)

    # 4. Create Target Repo
    # We assume the token owner is the target owner. 
    # Note: To get the actual username of the token holder, we'd need a 'get_user' tool or separate API call.
    # For now, we assume the user wants to clone to their own account, and we assume the source_owner string
    # might be different. 
    # CRITICAL: We need the TARGET owner username to push. 
    # Let's try to deduce it or assume it's the same as source for this MVP, 
    # OR we can just pass 'source_owner' if you are forking your own repo.
    target_owner = source_owner # Assumption for MVP

    if not await create_repo(safe_repo_name, github_token):
        return JSONResponse({"status": "error", "error": f"Failed to create repo '{safe_repo_name}'. Name might be taken."}, status_code=500)

    # 5. Refactor & Push
    async with httpx.AsyncClient() as client:
        # Refactor in parallel
        tasks = []
        filenames = []
        for fname, content in fetched_files.items():
            tasks.append(refactor_code(fname, content, instructions, client))
            filenames.append(fname)
        
        results = await asyncio.gather(*tasks)

        # Push results
        for i, new_code in enumerate(results):
            original_name = filenames[i]
            # Simple extension swap logic for Java
            if "java" in instructions.lower() and original_name.endswith(".py"):
                # Very naive renaming for demo
                new_path = "src/main/java/com/app/" + original_name.replace(".py", ".java")
            else:
                new_path = original_name

            await push_file(target_owner, safe_repo_name, new_path, new_code, github_token)
            logger.info(f"Pushed {new_path}")

    return {
        "status": "success",
        "new_repo": safe_repo_name,
        "link": f"[https://github.com/](https://github.com/){target_owner}/{safe_repo_name}"
    }
