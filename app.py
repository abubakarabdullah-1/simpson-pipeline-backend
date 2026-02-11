from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Dict
import fitz  # PyMuPDF
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime, timedelta
import uuid
import os
import shutil
import json
import traceback
import asyncio
import boto3
from botocore.exceptions import ClientError as BotoClientError

from dotenv import load_dotenv
from pymongo import MongoClient

from pipeline.runner import run_pipeline
from pipeline.excel_exporter import create_excel_from_result
from auth import get_current_user
from s3_utils import upload_pipeline_outputs, get_s3_client

# ... (omitted sections)
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

import dns.resolver

dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8', '8.8.4.4']

client = MongoClient(
    MONGO_URL,
    serverSelectionTimeoutMS=30000,  # Increased timeout to 30s
    connectTimeoutMS=30000,
    socketTimeoutMS=30000,
    retryWrites=True,
    retryReads=True
)

# Test connection at startup
try:
    client.admin.command('ping')
    print("✅ MongoDB connected successfully")
except Exception as e:
    print(f"⚠️ MongoDB connection warning: {e}")
    print("The app will start but database operations may fail")

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


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Simpson Pipeline Backend")

# -----------------------------
# CORS......
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://simpson.neuralogicgroup.com",
        "https://www.simpson.neuralogicgroup.com",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Pipeline Failure Detection
# -----------------------------
@app.on_event("startup")
async def startup_recovery():
    """Mark abandoned pipelines as FAILED on server startup"""
    result = runs_collection.update_many(
        {"status": {"$in": ["RUNNING", "IN_PROGRESS"]}},
        {
            "$set": {
                "status": "FAILED",
                "error": "Server was restarted while pipeline was running",
                "failed_at": datetime.utcnow()
            }
        }
    )
    if result.modified_count > 0:
        print(f"⚠️ Marked {result.modified_count} abandoned pipelines as FAILED")
    
    # Start timeout monitor
    asyncio.create_task(monitor_timeouts())


async def monitor_timeouts():
    """Background task to check for timed-out pipelines with retry logic"""
    timeout_minutes = 3  # timeout: 3 minutes (increased from 1 to handle longer processing)
    max_retries = 2  # Number of retry attempts before marking as FAILED

    while True:
        await asyncio.sleep(60)  # Check every minute
        
        timeout_threshold = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        
        # Use cursor to avoid loading all documents into RAM
        timed_out_pipelines = runs_collection.find(
            {
                "status": {"$in": ["RUNNING", "IN_PROGRESS"]},
                "last_updated": {"$lt": timeout_threshold}
            },
            {
                "run_id": 1, 
                "retry_count": 1, 
                "started_at": 1, 
                "last_updated": 1,
                "pdf_file": 1
            }
        )
        
        failed_count = 0
        for pipeline in timed_out_pipelines:
            run_id = pipeline["run_id"]
            retry_count = pipeline.get("retry_count", 0)
            started_at = pipeline.get("started_at")
            last_updated = pipeline.get("last_updated")
            pdf_file = pipeline.get("pdf_file", "unknown")
            
            # Calculate elapsed time
            if started_at:
                elapsed = datetime.utcnow() - started_at
                elapsed_str = f"{elapsed.total_seconds() / 60:.1f} minutes"
            else:
                elapsed_str = "unknown"
            
            # Calculate idle time (time since last update)
            if last_updated:
                idle_time = datetime.utcnow() - last_updated
                idle_str = f"{idle_time.total_seconds() / 60:.1f} minutes"
            else:
                idle_str = "unknown"
            
            if retry_count < max_retries:
                # Increment retry count and reset last_updated
                runs_collection.update_one(
                    {"run_id": run_id},
                    {
                        "$set": {"last_updated": datetime.utcnow()},
                        "$inc": {"retry_count": 1}
                    }
                )
                print(f"⚠️ Pipeline {run_id} timed out - retry {retry_count + 1}/{max_retries} (idle: {idle_str})")
            else:
                # Max retries exceeded - mark as FAILED with detailed error
                error_message = (
                    f"Pipeline timed out after {max_retries} retry attempts. "
                    f"Total runtime: {elapsed_str}, "
                    f"idle time: {idle_str} (threshold: {timeout_minutes} min). "
                    f"No activity detected since {last_updated.strftime('%Y-%m-%d %H:%M:%S UTC') if last_updated else 'unknown'}. "
                    f"This usually indicates the pipeline crashed or hung during processing."
                )
                
                runs_collection.update_one(
                    {"run_id": run_id},
                    {
                        "$set": {
                            "status": "FAILED",
                            "error": error_message,
                            "failed_at": datetime.utcnow(),
                            "timeout_details": {
                                "elapsed_time": elapsed_str,
                                "idle_time": idle_str,
                                "timeout_threshold_minutes": timeout_minutes,
                                "last_updated": last_updated.isoformat() if last_updated else None,
                                "started_at": started_at.isoformat() if started_at else None,
                            }
                        }
                    }
                )
                failed_count += 1
                print(f"❌ Pipeline {run_id} FAILED: {error_message}")
        
        if failed_count > 0:
            print(f"❌ Marked {failed_count} pipelines as FAILED after {max_retries} retries")



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
# Trigger Pipeline - Unified Endpoint
# -----------------------------
@app.post("/pipeline/run")
async def trigger_pipeline(
    background: BackgroundTasks,
    pdfs: List[UploadFile] = File(...),
    user: Dict = Depends(get_current_user)
):
    
    if not pdfs or len(pdfs) == 0:
        raise HTTPException(
            status_code=400, 
            detail="At least one PDF file is required"
        )

    run_id = str(uuid.uuid4())
    
    # Handle single file upload
    if len(pdfs) == 1:
        filename = f"{run_id}.pdf"
        file_path = os.path.join(UPLOAD_DIR, filename)

        # Save the single PDF file
        pdf_bytes = await pdfs[0].read()
        with open(file_path, "wb") as f:
            f.write(pdf_bytes)

        doc = {
            "run_id": run_id,
            "status": "RUNNING",
            "pdf_file": file_path,
            "pdf_count": 1,
            "started_at": datetime.utcnow(),
            "last_updated": datetime.utcnow(),
            "retry_count": 0,
        }

        try:
            runs_collection.insert_one(doc)
            background.add_task(run_and_store, run_id, file_path)
        except Exception as e:
            # Clean up the uploaded file if DB insert fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=503,
                detail=f"Database connection error: {str(e)}. Please check your MongoDB connection."
            )

        return {
            "run_id": run_id,
            "status": "started",
            "pdf_count": 1,
            "message": "Single PDF processing started"
        }
    
    # Handle multiple file uploads
    else:
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
            "last_updated": datetime.utcnow(),
            "retry_count": 0,
        }

        try:
            runs_collection.insert_one(doc)
            background.add_task(run_and_store, run_id, file_path)
        except Exception as e:
            # Clean up the merged file if DB insert fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=503,
                detail=f"Database connection error: {str(e)}. Please check your MongoDB connection."
            )

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
        # MARK AS COMPLETED FIRST
        # ------------------
        # Mark pipeline as COMPLETED immediately after processing
        # This prevents S3 upload issues from incorrectly marking the pipeline as failed
        
        mongo_payload = {
            "status": "COMPLETED",
            "result": result,
            "result_file": json_path,
            "excel_file": excel_path,
            "debug_pdf": result.get("debug_pdf"),
            "log_file": result.get("log_file"),
            "ended_at": datetime.utcnow(),
            "confidence": result.get("confidence"),
            "s3_upload_status": "IN_PROGRESS",
            "s3_data": {},
        }

        safe_payload = stringify_keys(mongo_payload)

        runs_collection.update_one(
            {"run_id": run_id},
            {"$set": safe_payload},
        )

        # ------------------
        # Upload to S3 (non-blocking for pipeline completion)
        # ------------------
        files_to_upload = {
            "excel": excel_path,
            "json": json_path,
            "pdf_original": pdf_path,  # Upload original PDF so it can be viewed later
        }
        
        # Add optional files if they exist
        if result.get("debug_pdf"):
            files_to_upload["debug_pdf"] = result.get("debug_pdf")
        
        if result.get("log_file"):
            files_to_upload["log_file"] = result.get("log_file")
        
        # Upload all files to S3 in parallel
        try:
            s3_data = upload_pipeline_outputs(run_id, files_to_upload, cleanup_local=True)
            
            # Update MongoDB with S3 URLs after successful upload
            runs_collection.update_one(
                {"run_id": run_id},
                {"$set": {
                    "s3_data": stringify_keys(s3_data),
                    "s3_upload_status": "COMPLETED"
                }},
            )
            
            # Clean up the uploaded PDF file after successful S3 upload
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    print(f"🗑️  Deleted uploaded PDF: {pdf_path}")
            except Exception as cleanup_exc:
                print(f"⚠️ Could not delete uploaded PDF {pdf_path}: {cleanup_exc}")
                
        except Exception as s3_exc:
            # S3 upload failed, but pipeline itself succeeded
            print(f"⚠️ S3 upload failed for run {run_id}: {s3_exc}")
            runs_collection.update_one(
                {"run_id": run_id},
                {"$set": {
                    "s3_upload_status": "FAILED",
                    "s3_error": str(s3_exc)
                }},
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


# -----------------------------
# Public File Download Endpoints (S3 Streaming)
# -----------------------------

def stream_from_s3(s3_key: str, filename: str):
    """Helper to stream file from S3 to client"""
    try:
        s3 = get_s3_client()
        bucket = os.getenv("S3_BUCKET_NAME")
        
        # Get object from S3
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        
        return StreamingResponse(
            obj['Body'].iter_chunks(),
            media_type=obj.get('ContentType', 'application/octet-stream'),
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except BotoClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            raise HTTPException(status_code=404, detail="File not found in S3")
        print(f"S3 Streaming Error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving file from storage")


@app.get("/outputs/{filename}")
def download_output(filename: str):
    """Download output files (JSON, Excel, debug PDFs) from S3"""
    # Extract run_id from filename (e.g. "abc-123.json" -> "abc-123")
    # Handles: run_id.ext, run_id_debug.pdf
    run_id = filename.split('.')[0].split('_')[0]
    
    s3_key = f"pipeline-outputs/{run_id}/{filename}"
    return stream_from_s3(s3_key, filename)


@app.get("/logs/{filename}")
def download_log(filename: str):
    """Download log files from S3"""
    # Extract run_id from filename (e.g. "abc-123_logs.json" -> "abc-123")
    run_id = filename.split('.')[0].split('_')[0]
    
    s3_key = f"pipeline-outputs/{run_id}/{filename}"
    return stream_from_s3(s3_key, filename)


@app.get("/error_logs/{filename}")
def download_error_log(filename: str):
    """Download error log files (Local only)"""
    file_path = os.path.join(ERROR_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.get("/uploads/{filename}")
def download_upload(filename: str):
    """
    Download uploaded PDF files.
    1. Check local storage first (in case it hasn't been cleaned up yet).
    2. If missing locally, stream from S3 (original PDF is uploaded as 'pdf_original').
    """
    # 1. Try local file
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
        
    # 2. Try S3 (Fallback)
    # Filename format is usually "{run_id}.pdf" or "{run_id}_merged.pdf"
    try:
        run_id = filename.split('.')[0].split('_')[0]
        s3_key = f"pipeline-outputs/{run_id}/{filename}"
        return stream_from_s3(s3_key, filename)
    except Exception:
        raise HTTPException(status_code=404, detail="File not found locally or in S3")
