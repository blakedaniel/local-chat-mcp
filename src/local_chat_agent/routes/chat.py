"""Chat and refactoring endpoints."""

import os
import re
import shutil
import tempfile
import zipfile
import asyncio

import httpx
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..config import settings

router = APIRouter()

ALLOWED_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css',
    '.java', '.cpp', '.c', '.h', '.rs', '.go', '.php', '.rb',
    '.json', '.yaml', '.yml', '.sql', '.md', '.txt',
}


def parse_and_save_files(raw_text: str, base_dir: str):
    """Scan for '### FILE: <name>' markers and save multiple files.

    Returns list of created filenames, or None if no markers found.
    """
    segments = re.split(r'### FILE:\s*([^\n]+)\n', raw_text)

    if len(segments) < 3:
        return None

    created_files = []
    for i in range(1, len(segments), 2):
        fname = segments[i].strip()
        content = segments[i + 1]

        # Cleanup markdown code blocks
        if content.strip().startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)

        # Prevent path traversal
        if ".." in fname or fname.startswith("/") or fname.startswith("\\"):
            print(f"Skipping unsafe filename: {fname}")
            continue

        full_path = os.path.join(base_dir, fname)
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
    mcp_manager=None,
) -> dict:
    """Process a single file through the LLM with optional RAG context.

    Returns a dict with keys: filename, status ("success", "error", "skipped"), error (optional).
    """
    filename = os.path.basename(file_path)
    if filename.startswith("._") or filename == ".DS_Store":
        return {"filename": filename, "status": "skipped", "error": "macOS metadata file"}
    if os.path.splitext(filename)[1].lower() not in ALLOWED_EXTENSIONS:
        return {"filename": filename, "status": "skipped", "error": "unsupported extension"}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[process_file] Could not read {filename}: {e}")
        return {"filename": filename, "status": "error", "error": f"could not read file: {e}"}

    print(f"[process_file] Using Ollama at {settings.ollama_url} with model {settings.model_name}")

    # Consult RAG if MCP tools available
    rag_context = ""
    if mcp_manager and mcp_manager.get_all_tools():
        try:
            rag_query = f"{user_instructions} {filename}"
            print(f"[RAG] Querying knowledge base: {rag_query[:100]}...")
            result = await mcp_manager.call_tool_by_name("query", {"question": rag_query})

            if hasattr(result, 'content'):
                rag_text = "\n".join(
                    item.text if hasattr(item, 'text') else str(item)
                    for item in result.content
                )
            else:
                rag_text = str(result)

            if rag_text and "error" not in rag_text.lower():
                rag_context = (
                    "\n\n--- KNOWLEDGE BASE CONTEXT ---\n"
                    "The following information was retrieved from the knowledge base. "
                    "Use this context to inform your code generation:\n\n"
                    f"{rag_text}\n"
                    "--- END CONTEXT ---\n"
                )
                print(f"[RAG] Retrieved context ({len(rag_text)} chars)")
            else:
                print("[RAG] No relevant context found")
        except Exception as e:
            print(f"[RAG] Query failed: {e}")

    # Build prompts
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

    system_prompt = base_system_prompt + rag_context

    user_prompt = (
        f"CURRENT FILE: {filename}\n"
        f"INSTRUCTION: {user_instructions}\n"
        f"CONTENT:\n{content}"
    )

    try:
        response = await client.post(
            settings.ollama_url,
            json={
                "model": settings.model_name,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 16384,
                },
            },
            timeout=None,
        )
        response.raise_for_status()
        raw_output = response.json().get("response", "")
        print(f"[process_file] {filename}: Ollama returned {len(raw_output)} chars")

        # Try multi-file output
        new_files = parse_and_save_files(raw_output, extract_root)

        if new_files:
            print(f"Converted {filename} into {len(new_files)} new files:")
            for nf in new_files:
                print(f"   -> {nf}")
            os.remove(file_path)
            print(f"   (Removed original {filename})")
            return {"filename": filename, "status": "success"}
        else:
            # Single-file refactor
            cleaned_code = raw_output.strip()
            if cleaned_code.startswith("```"):
                cleaned_code = cleaned_code.split("\n", 1)[1].rsplit("\n", 1)[0]
            if len(cleaned_code) > 0 and "I cannot assist" not in cleaned_code:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_code)
                print(f"Refactored {filename} (Single file update)")
                return {"filename": filename, "status": "success"}
            else:
                print(f"[process_file] {filename}: LLM returned empty or refused response. Raw output length: {len(raw_output)}, starts with: {repr(raw_output[:200])}")
                return {"filename": filename, "status": "error", "error": "LLM returned empty or refused response"}

    except Exception as e:
        import traceback
        print(f"Error processing {filename}: {type(e).__name__} - {e}")
        traceback.print_exc()
        return {"filename": filename, "status": "error", "error": f"{type(e).__name__}: {e}"}


@router.post("/refactor")
async def refactor_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    instructions: str = Form(...),
):
    print(f"[refactor] Request received. Ollama URL: {settings.ollama_url}, Model: {settings.model_name}")

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
            for fname in files:
                files_to_process.append(os.path.join(root, fname))

        sem = asyncio.Semaphore(1)
        results: list[dict] = []

        async def worker(fp, instr, client):
            async with sem:
                result = await process_file(fp, instr, client, extract_dir)
                if result:
                    results.append(result)

        async with httpx.AsyncClient() as client:
            tasks = [worker(fp, instructions, client) for fp in files_to_process]
            await asyncio.gather(*tasks)

        # Check results: if every processable file failed, return an error instead of unchanged files
        processed = [r for r in results if r["status"] != "skipped"]
        failed = [r for r in processed if r["status"] == "error"]

        if processed and len(failed) == len(processed):
            error_details = [
                {"file": r["filename"], "error": r.get("error", "unknown")}
                for r in failed
            ]
            print(f"[refactor] All {len(failed)} file(s) failed processing")
            return JSONResponse(
                status_code=502,
                content={
                    "error": "All files failed to process. Is Ollama running and the model pulled?",
                    "details": error_details,
                },
            )

        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(extract_dir):
                for fname in files:
                    file_path = os.path.join(root, fname)
                    arcname = os.path.relpath(file_path, extract_dir)
                    z.write(file_path, arcname)

        # Log summary
        succeeded = [r for r in processed if r["status"] == "success"]
        print(f"[refactor] Done: {len(succeeded)} succeeded, {len(failed)} failed, {len(results) - len(processed)} skipped")

        return FileResponse(output_zip, filename="converted_project.zip", media_type="application/zip")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        background_tasks.add_task(shutil.rmtree, work_dir)
