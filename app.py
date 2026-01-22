from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from datetime import datetime
import uuid
import os
import shutil

from pymongo import MongoClient

from pipeline.runner import run_pipeline


# -----------------------------
# Mongo Setup
# -----------------------------
MONGO_URL = "mongodb://localhost:27017"
client = MongoClient(MONGO_URL)

db = client["simpson_pipeline"]
runs_collection = db["runs"]


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Simpson Pipeline Backend")


UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Trigger Pipeline
# -----------------------------
@app.post("/pipeline/run")
async def trigger_pipeline(
    background: BackgroundTasks,
    pdf: UploadFile = File(...)
):

    run_id = str(uuid.uuid4())

    file_path = os.path.join(UPLOAD_DIR, f"{run_id}_{pdf.filename}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)

    doc = {
        "run_id": run_id,
        "status": "RUNNING",
        "pdf_file": file_path,
        "started_at": datetime.utcnow(),
    }

    runs_collection.insert_one(doc)

    background.add_task(run_and_store, run_id, file_path)

    return {"run_id": run_id, "status": "started"}


# -----------------------------
# Worker
# -----------------------------
def run_and_store(run_id: str, pdf_path: str):

    try:
        result = run_pipeline(pdf_path)

        # Save JSON result
        json_path = os.path.join(OUTPUT_DIR, f"{run_id}.json")
        with open(json_path, "w") as f:
            import json
            json.dump(result, f, indent=2)

        runs_collection.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "status": "COMPLETED",
                    "result": result,
                    "result_file": json_path,
                    "ended_at": datetime.utcnow(),
                }
            },
        )

    except Exception as exc:

        runs_collection.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "status": "FAILED",
                    "error": str(exc),
                    "ended_at": datetime.utcnow(),
                }
            },
        )


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
