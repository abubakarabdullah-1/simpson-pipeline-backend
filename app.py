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
import threading
import boto3
from botocore.exceptions import ClientError as BotoClientError

from dotenv import load_dotenv
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor

from pipeline.runner import run_pipeline
from pipeline import Fascia_Gemini, Reveal_Gemini
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
app = FastAPI(
    title="Simpson Pipeline Backend",
    #docs_url=None,     # Disable Swagger UI
    #redoc_url=None,    # Disable Redoc
    #openapi_url=None   # Disable OpenAPI schema (completely hides /openapi.json)
)

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


@app.on_event("shutdown")
async def shutdown_cleanup():
    """Mark any still-running pipelines as FAILED when the server shuts down (Ctrl+C etc.)"""
    try:
        result = runs_collection.update_many(
            {"status": {"$in": ["RUNNING", "IN_PROGRESS"]}},
            {
                "$set": {
                    "status": "FAILED",
                    "error": "Server was stopped while pipeline was running",
                    "failed_at": datetime.utcnow()
                }
            }
        )
        if result.modified_count > 0:
            print(f"⏹️ Shutdown: marked {result.modified_count} pipeline(s) as FAILED")
    except Exception as e:
        print(f"⚠️ Shutdown cleanup failed: {e}")


async def monitor_timeouts():
    """Background task to check for timed-out pipelines with retry logic.
    Never crashes — all MongoDB errors are caught so the loop keeps running."""
    timeout_minutes = 60  # pipelines can legitimately run this long (Fascia+Reveal+Main)
    max_retries = 3  # Number of retry attempts before marking as FAILED

    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
        except asyncio.CancelledError:
            # Server is shutting down — exit cleanly
            print("⏹️  monitor_timeouts: server shutting down, exiting.")
            return

        try:
            timeout_threshold = datetime.utcnow() - timedelta(minutes=timeout_minutes)

            # Materialise the cursor to a list so no cursor stays open
            # while we do further writes (avoids cursor-level cancellations)
            timed_out_pipelines = list(runs_collection.find(
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
            ))

            failed_count = 0
            for pipeline in timed_out_pipelines:
                try:
                    run_id = pipeline["run_id"]
                    retry_count = pipeline.get("retry_count", 0)
                    started_at = pipeline.get("started_at")
                    last_updated = pipeline.get("last_updated")

                    elapsed_str = (
                        f"{(datetime.utcnow() - started_at).total_seconds() / 60:.1f} minutes"
                        if started_at else "unknown"
                    )
                    idle_str = (
                        f"{(datetime.utcnow() - last_updated).total_seconds() / 60:.1f} minutes"
                        if last_updated else "unknown"
                    )

                    if retry_count < max_retries:
                        runs_collection.update_one(
                            {"run_id": run_id},
                            {
                                "$set": {"last_updated": datetime.utcnow()},
                                "$inc": {"retry_count": 1}
                            }
                        )
                        print(f"⚠️ Pipeline {run_id} timed out - retry {retry_count + 1}/{max_retries} (idle: {idle_str})")
                    else:
                        error_message = (
                            f"Pipeline timed out after {max_retries} retry attempts. "
                            f"Total runtime: {elapsed_str}, idle time: {idle_str} "
                            f"(threshold: {timeout_minutes} min). "
                            f"No activity detected since "
                            f"{last_updated.strftime('%Y-%m-%d %H:%M:%S UTC') if last_updated else 'unknown'}."
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

                except Exception as per_pipeline_exc:
                    print(f"⚠️ monitor_timeouts: error processing pipeline record: {per_pipeline_exc}")
                    continue

            if failed_count > 0:
                print(f"❌ Marked {failed_count} pipelines as FAILED after {max_retries} retries")

        except asyncio.CancelledError:
            print("⏹️  monitor_timeouts: cancelled during DB check, exiting.")
            return
        except Exception as exc:
            # DB unavailable, network blip, etc. — log and wait for next cycle
            print(f"⚠️ monitor_timeouts cycle error (will retry next minute): {exc}")




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
    # user: Dict = Depends(get_current_user)  # Temporarily bypassed for testing
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
        # Initialize pipeline runner heartbeat collection + cancel event
        import pipeline.runner
        pipeline.runner.runs_collection = runs_collection
        pipeline.runner.cancel_event = threading.Event()  # fresh event per run
        _cancel = pipeline.runner.cancel_event

        # Master heartbeat thread — covers ALL 3 pipelines continuously
        _hb_stop = threading.Event()
        def _heartbeat_loop():
            while not _hb_stop.is_set():
                pipeline.runner.update_heartbeat(run_id)
                _hb_stop.wait(15)  # heartbeat every 15s for safety
        _hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        _hb_thread.start()

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_main = executor.submit(run_pipeline, pdf_path, run_id)
            future_fascia = executor.submit(Fascia_Gemini.run_full_document, pdf_path, run_id)
            future_reveal = executor.submit(Reveal_Gemini.run_full_document, pdf_path, run_id)
            
            futures = {
                future_main: "Main",
                future_fascia: "Fascia",
                future_reveal: "Reveal",
            }

            # Wait for all futures; append annotated pages as each pipeline completes
            from concurrent.futures import as_completed
            from pipeline.debug_pdf_collector import append_annotated_to_debug_pdf
            results_map = {}
            current_debug_pdf = None       # set when Main finishes
            pending_annotated_entries = [] # entries from Fascia/Reveal that arrived before Main

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result_data = future.result()
                    results_map[name] = result_data
                    print(f"✅ {name} pipeline completed for {run_id}")

                    if name == "Main":
                        # Main sets the base debug PDF path
                        current_debug_pdf = (result_data or {}).get("debug_pdf")
                        # If Fascia/Reveal already finished, append their queued entries now
                        if pending_annotated_entries and current_debug_pdf:
                            print(f"[DEBUG-PDF] Main done — appending {len(pending_annotated_entries)} queued annotated entries")
                            updated = append_annotated_to_debug_pdf(
                                existing_pdf_path=current_debug_pdf,
                                annotated_entries=pending_annotated_entries,
                                output_dir=OUTPUT_DIR,
                                run_id=run_id,
                            )
                            if updated:
                                current_debug_pdf = updated
                                result_data["debug_pdf"] = updated
                            pending_annotated_entries = []

                    elif name in ("Fascia", "Reveal"):
                        entries = (result_data or {}).get("annotated_entries") or []
                        if entries:
                            if current_debug_pdf:
                                # Main already finished — append immediately
                                print(f"[DEBUG-PDF] {name} done — appending {len(entries)} annotated entries")
                                updated = append_annotated_to_debug_pdf(
                                    existing_pdf_path=current_debug_pdf,
                                    annotated_entries=entries,
                                    output_dir=OUTPUT_DIR,
                                    run_id=run_id,
                                )
                                if updated:
                                    current_debug_pdf = updated
                                    # Keep Main result's debug_pdf in sync
                                    main_res = results_map.get("Main")
                                    if isinstance(main_res, dict):
                                        main_res["debug_pdf"] = updated
                            else:
                                # Main not finished yet — queue entries for later
                                print(f"[DEBUG-PDF] {name} done — queuing {len(entries)} entries (waiting for Main's debug PDF)")
                                pending_annotated_entries.extend(entries)

                except Exception as e:
                    print(f"❌ {name} pipeline failed for {run_id}: {e}")
                    results_map[name] = None
                    # Stop heartbeat FIRST so no MongoDB ops are in-flight when
                    # we set the cancel event (otherwise PyMongo raises _OperationCancelled)
                    _hb_stop.set()
                    _hb_thread.join(timeout=3)
                    # Now safe to signal cancellation to remaining pipeline threads
                    _cancel.set()
                    print(f"🛑 Cancel signal sent — stopping remaining pipelines")

            # Safety: if Main never produced a debug PDF but Fascia/Reveal did, create one
            if pending_annotated_entries:
                print(f"[DEBUG-PDF] Appending {len(pending_annotated_entries)} remaining queued entries")
                updated = append_annotated_to_debug_pdf(
                    existing_pdf_path=current_debug_pdf,
                    annotated_entries=pending_annotated_entries,
                    output_dir=OUTPUT_DIR,
                    run_id=run_id,
                )
                if updated:
                    current_debug_pdf = updated
                    main_res = results_map.get("Main")
                    if isinstance(main_res, dict):
                        main_res["debug_pdf"] = updated

        # Stop heartbeat thread (idempotent — safe to call even if already stopped)
        _hb_stop.set()
        _hb_thread.join(timeout=5)

        # Clear cancel event so MongoDB writes in the main thread below are not affected
        _cancel.clear()
        import time as _time; _time.sleep(0.2)  # brief pause for any lingering thread ops


        # Build result dicts from collected results
        result = results_map.get("Main")
        if result is None:
            result = {
                "status": "FAILED", "error": "Main pipeline failed",
                "project_specs": {}, "survey_data": {}, "scale_data": [],
                "line_items": [], "grand_total": 0, "logs": ["Main pipeline failed"],
                "confidence": 0, "debug_pdf": None, "log_file": None
            }

        fascia_result = results_map.get("Fascia")
        if fascia_result is None:
            fascia_result = {"status": "FAILED", "error": "Fascia pipeline failed or cancelled"}

        reveal_result = results_map.get("Reveal")
        if reveal_result is None:
            reveal_result = {"status": "FAILED", "error": "Reveal pipeline failed or cancelled"}
            
        result["fascia_extraction"] = fascia_result
        result["reveal_extraction"] = reveal_result



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
        # Add Fascia & Reveal to Logs
        # ------------------
        log_file_path = result.get("log_file")
        if log_file_path and os.path.exists(log_file_path):
            try:
                with open(log_file_path, "r") as f:
                    logs_data = json.load(f)
                
                # Helper to inject extraction results into logs
                def _inject_extraction_to_logs(extraction, source_name):
                    # Ensure every page has the source_name key initialized to []
                    for page_num_str in logs_data.get("pages", {}):
                        if source_name not in logs_data["pages"][page_num_str]:
                            logs_data["pages"][page_num_str][source_name] = []

                    if not extraction or not isinstance(extraction, dict): return
                    for page_entry in extraction.get("page_results", []):
                        if page_entry.get("result", {}).get("status") != "SUCCESS": continue
                        
                        page_num_str = str(page_entry.get("page", 0) + 1)
                        if page_num_str not in logs_data.get("pages", {}):
                            logs_data.setdefault("pages", {})[page_num_str] = {}
                            logs_data["pages"][page_num_str][source_name] = []
                        
                        page_logs = logs_data["pages"][page_num_str]
                            
                        keyword = page_entry.get("keyword", source_name)
                        for occ in page_entry.get("result", {}).get("occurrence_results", []):
                            p2 = occ.get("phase2", {}) or {}
                            p3 = occ.get("phase3", {}) or {}
                            p1b = page_entry.get("result", {}).get("phase1b", {}) or {}
                            
                            log_entry = {
                                "keyword": keyword,
                                "view": p2.get("drawing_title", "Unknown View"),
                                "material": p3.get("material", "Unknown Material"),
                            }
                            
                            # Add dimension if available
                            dim = p3.get("dimension_label_text")
                            if dim: log_entry["dimension"] = dim

                            # Add phase 1b description if available
                            desc = p1b.get("description")
                            if desc: log_entry["description"] = desc
                                
                            page_logs[source_name].append(log_entry)

                _inject_extraction_to_logs(result.get("fascia_extraction"), "Fascia")
                _inject_extraction_to_logs(result.get("reveal_extraction"), "Reveal")

                with open(log_file_path, "w") as f:
                    json.dump(logs_data, f, indent=2)
            except Exception as e:
                print(f"⚠️ Failed to inject Fascia/Reveal into logs: {e}")

        # ------------------
        # MARK AS COMPLETED FIRST
        # ------------------
        # Mark pipeline as COMPLETED only if at least one component finished without exception
        # or if they all failed but were handled. 
        # Actually, if we reach here, we have a result for all three (even if it's error dict).
        
        all_failed = (result.get("status") == "FAILED" and 
                     fascia_result.get("status") == "FAILED" and 
                     reveal_result.get("status") == "FAILED")

        # ------------------
        # Split Debug PDF into individual pages
        # ------------------
        debug_pdf_path = result.get("debug_pdf")
        debug_pdf_dir = None
        if debug_pdf_path and os.path.exists(debug_pdf_path):
            try:
                import fitz
                debug_pdf_dir = os.path.join(OUTPUT_DIR, run_id, "pdf")
                os.makedirs(debug_pdf_dir, exist_ok=True)
                
                doc = fitz.open(debug_pdf_path)
                for i in range(len(doc)):
                    page_pdf_path = os.path.join(debug_pdf_dir, f"{run_id}_page_{i+1}.pdf")
                    # Create a new empty PDF for this single page
                    new_doc = fitz.open()
                    new_doc.insert_pdf(doc, from_page=i, to_page=i)
                    new_doc.save(page_pdf_path)
                    new_doc.close()
                doc.close()
                print(f"[DEBUG-PDF] Split {len(doc)} pages into {debug_pdf_dir}")
            except Exception as e:
                print(f"⚠️ Failed to split debug PDF: {e}")

        mongo_payload = {
            "status": "FAILED" if all_failed else "COMPLETED",
            "result": result,
            "result_file": json_path,
            "excel_file": excel_path,
            "debug_pdf": debug_pdf_path,
            "debug_pdf_dir": debug_pdf_dir,
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
        print(f"✅ Pipeline {run_id} finished")

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
            
        # NOTE: Fascia/Reveal annotated pages are now embedded directly into the debug PDF.
        # No separate annotated PDF files are generated.

        # Upload all files to S3 in parallel
        try:
            s3_data = upload_pipeline_outputs(run_id, files_to_upload, cleanup_local=True)

            # Update MongoDB with S3 URLs after successful upload
            runs_collection.update_one(
                {"run_id": run_id},
                {"$set": {
                    "result": stringify_keys(result), # Updated result with S3 URLs
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
    
    # 1. Try to extract run_id from filename (Standard way)
    # Handles: run_id.ext, run_id_debug.pdf
    run_id = filename.split('.')[0].split('_')[0]
    
    # 2. If it looks like a timestamp (e.g. starts with "202..."), lookup via Mongo
    # Because old debug PDFs are named "YYYYMMDD_HHMMSS_debug.pdf" and don't contain run_id
    if filename.startswith("202") or filename.startswith("203"):
        print(f"🔍 Legacy filename detected: {filename}, looking up in MongoDB...")
        run = runs_collection.find_one({"result.debug_pdf": {"$regex": filename}})
        if run:
            run_id = run["run_id"]
            print(f"✅ Found run_id for legacy file: {run_id}")
        else:
            print(f"❌ Could not find run_id for file: {filename}")
    
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
