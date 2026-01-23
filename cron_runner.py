import os
import glob
from datetime import datetime
import json

from pymongo import MongoClient

from pipeline.runner import run_pipeline

# -----------------------------
# Mongo Setup (MongoDB Atlas)
# -----------------------------
MONGO_URL = (
    "mongodb+srv://abubakarabdullah:DtZ1DAu5IMp83rVI@cluster0.clmrsac.mongodb.net/?appName=Cluster0"
)

client = MongoClient(MONGO_URL)

db = client["simpson_pipeline"]
runs_collection = db["runs"]

# -----------------------------
# Mongo Setup (MongoDB Atlas)
# -----------------------------
MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    raise RuntimeError("MONGO_URL environment variable is not set")

client = MongoClient(MONGO_URL)

db = client["simpson_pipeline"]
runs_collection = db["runs"]


# -----------------------------
# Settings
# -----------------------------
INPUT_DIR = "cron_inputs"
ARCHIVE_DIR = "cron_archive"
OUTPUT_DIR = "outputs"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Batch Processor
# -----------------------------
def run_batch():

    pdfs = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))

    print(f"[CRON] Found {len(pdfs)} PDFs")

    for pdf_path in pdfs:

        filename = os.path.basename(pdf_path)

        run_id = f"cron-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{filename}"

        doc = {
            "run_id": run_id,
            "status": "RUNNING",
            "pdf_file": pdf_path,
            "started_at": datetime.utcnow(),
            "trigger": "cron",
        }

        runs_collection.insert_one(doc)

        try:
            result = run_pipeline(pdf_path)

            # ------------------
            # Save JSON result
            # ------------------
            json_path = os.path.join(OUTPUT_DIR, f"{run_id}.json")

            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)

            # ------------------
            # Update Mongo
            # ------------------
            runs_collection.update_one(
                {"run_id": run_id},
                {
                    "$set": {
                        "status": "COMPLETED",
                        "result": result,
                        "result_file": json_path,
                        "ended_at": datetime.utcnow(),
                        "confidence": result.get("confidence"),
                    }
                },
            )

            # ------------------
            # Archive PDF
            # ------------------
            os.rename(
                pdf_path,
                os.path.join(ARCHIVE_DIR, filename),
            )

            print(f"[CRON] Finished {filename}")

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

            print(f"[CRON] FAILED {filename}", exc)


if __name__ == "__main__":
    run_batch()
