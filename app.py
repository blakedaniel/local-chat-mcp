import os
import shutil
import zipfile
import tempfile
import asyncio
import httpx
import re
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
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

async def process_file(file_path: str, user_instructions: str, client: httpx.AsyncClient, extract_root: str):
    filename = os.path.basename(file_path)
    if os.path.splitext(filename)[1].lower() not in ALLOWED_EXTENSIONS:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return 

    # --- ADVANCED PROMPT FOR MULTI-FILE GENERATION ---
    system_prompt = (
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
        traceback.print_exc() # Prints the full stack trace to logs

# ... (Standard Boilerplate below) ...

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