from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from typing import List
import fitz  # PyMuPDF
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime
import uuid
import os
import shutil
import json
import traceback

from dotenv import load_dotenv
from pymongo import MongoClient

from pipeline.runner import run_pipeline
from pipeline.excel_exporter import create_excel_from_result


# -----------------------------
# Load ENV
# -----------------------------
load_dotenv()


# -----------------------------
# Mongo Setup (Atlas via .env)
# -----------------------------
MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    raise RuntimeError("MONGO_URL not set in .env file")

client = MongoClient(MONGO_URL)
db = client["simpson_pipeline"]
runs_collection = db["runs"]


# -----------------------------
# Directories
# -----------------------------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
ERROR_DIR = "error_logs"
LOGS_DIR = "logs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


from fastapi.staticfiles import StaticFiles

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Simpson Pipeline Backend")

# -----------------------------
# Static Mounts
# -----------------------------
app.mount(f"/{OUTPUT_DIR}", StaticFiles(directory=OUTPUT_DIR), name=OUTPUT_DIR)
app.mount(f"/{LOGS_DIR}", StaticFiles(directory=LOGS_DIR), name=LOGS_DIR)
app.mount(f"/{ERROR_DIR}", StaticFiles(directory=ERROR_DIR), name=ERROR_DIR)
app.mount(f"/{UPLOAD_DIR}", StaticFiles(directory=UPLOAD_DIR), name=UPLOAD_DIR)

# -----------------------------
# CORS......
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://simpson.neuralogicgroup.com",
        "https://www.simpson.neuralogicgroup.com",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Helpers
# -----------------------------
def stringify_keys(obj):
    if isinstance(obj, dict):
        return {str(k): stringify_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [stringify_keys(v) for v in obj]
    else:
        return obj


# -----------------------------
# Trigger Pipeline - Single PDF
# -----------------------------
@app.post("/pipeline/run/single")
async def trigger_pipeline_single(
    background: BackgroundTasks,
    pdf: UploadFile = File(...)
):
    """
    Process a single PDF file through the pipeline.
    """
    run_id = str(uuid.uuid4())
    filename = f"{run_id}.pdf"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Save the single PDF file
    pdf_bytes = await pdf.read()
    with open(file_path, "wb") as f:
        f.write(pdf_bytes)

    doc = {
        "run_id": run_id,
        "status": "RUNNING",
        "pdf_file": file_path,
        "started_at": datetime.utcnow(),
    }

    runs_collection.insert_one(doc)

    background.add_task(run_and_store, run_id, file_path)

    return {
        "run_id": run_id,
        "status": "started",
        "message": "Single PDF processing started"
    }


# -----------------------------
# Trigger Pipeline - Multiple PDFs
# -----------------------------
@app.post("/pipeline/run/multiple")
async def trigger_pipeline_multiple(
    background: BackgroundTasks,
    pdfs: List[UploadFile] = File(...)
):
    """
    Process multiple PDF files by merging them into a single PDF and running the pipeline.
    """
    if len(pdfs) < 2:
        raise HTTPException(
            status_code=400, 
            detail="Multiple endpoint requires at least 2 PDF files. Use /pipeline/run/single for a single file."
        )

    run_id = str(uuid.uuid4())
    merged_filename = f"{run_id}_merged.pdf"
    file_path = os.path.join(UPLOAD_DIR, merged_filename)

    # Merge multiple PDFs into a single PDF
    merged_doc = fitz.open()
    for pdf_file in pdfs:
        # Read bytes into memory
        pdf_bytes = await pdf_file.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as src_doc:
            merged_doc.insert_pdf(src_doc)
    
    merged_doc.save(file_path)
    merged_doc.close()

    doc = {
        "run_id": run_id,
        "status": "RUNNING",
        "pdf_file": file_path,
        "pdf_count": len(pdfs),
        "started_at": datetime.utcnow(),
    }

    runs_collection.insert_one(doc)

    background.add_task(run_and_store, run_id, file_path)

    return {
        "run_id": run_id,
        "status": "started",
        "pdf_count": len(pdfs),
        "message": f"Merged {len(pdfs)} PDFs and started processing"
    }


# -----------------------------
# Worker
# -----------------------------
def run_and_store(run_id: str, pdf_path: str):

    try:
        result = run_pipeline(pdf_path, run_id=run_id)

        # ------------------
        # Create Excel FIRST
        # ------------------
        excel_path = os.path.join(OUTPUT_DIR, f"{run_id}.xlsx")
        create_excel_from_result(result, excel_path)

        # inject path into JSON
        result["excel_file"] = excel_path

        # ------------------
        # Save JSON locally
        # ------------------
        json_path = os.path.join(OUTPUT_DIR, f"{run_id}.json")

        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        # ------------------
        # Mongo-safe payload
        # ------------------
        mongo_payload = {
            "status": "COMPLETED",
            "result": result,
            "result_file": json_path,
            "excel_file": excel_path,
            "debug_pdf": result.get("debug_pdf"),
            "log_file": result.get("log_file"),
            "ended_at": datetime.utcnow(),
            "confidence": result.get("confidence"),
        }

        safe_payload = stringify_keys(mongo_payload)

        runs_collection.update_one(
            {"run_id": run_id},
            {"$set": safe_payload},
        )

    except Exception as exc:

        tb = traceback.format_exc()

        # ------------------
        # Write local error log
        # ------------------
        error_path = os.path.join(ERROR_DIR, f"{run_id}.txt")

        with open(error_path, "w") as f:
            f.write(f"RUN ID: {run_id}\n")
            f.write(f"PDF: {pdf_path}\n")
            f.write(f"TIME: {datetime.utcnow()}\n\n")
            f.write(str(exc))
            f.write("\n\nTRACEBACK:\n")
            f.write(tb)

        # ------------------
        # Mongo-safe FAILED payload
        # ------------------
        mongo_payload = {
            "status": "FAILED",
            "error": str(exc),
            "traceback": tb,
            "error_file": error_path,
            "ended_at": datetime.utcnow(),
        }

        safe_payload = stringify_keys(mongo_payload)

        try:
            runs_collection.update_one(
                {"run_id": run_id},
                {"$set": safe_payload},
            )
        except Exception as mongo_exc:
            print("!!! FAILED TO WRITE ERROR TO MONGO !!!")
            print(mongo_exc)



# -----------------------------
# Status Endpoint
# -----------------------------
@app.get("/pipeline/{run_id}")
def get_pipeline_status(run_id: str):

    run = runs_collection.find_one(
        {"run_id": run_id},
        {"_id": 0},
    )

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return run
