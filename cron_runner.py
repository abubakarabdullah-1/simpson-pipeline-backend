import os
import glob
from datetime import datetime
import json
import traceback

from dotenv import load_dotenv
from pymongo import MongoClient

from pipeline.runner import run_pipeline
from pipeline.excel_exporter import create_excel_from_result


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
# Load ENV
# -----------------------------
load_dotenv()


# -----------------------------
# Mongo Setup (MongoDB Atlas via .env)
# -----------------------------
MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    raise RuntimeError("MONGO_URL not set in .env file")

client = MongoClient(MONGO_URL)

db = client["simpson_pipeline"]
runs_collection = db["runs"]


# -----------------------------
# Settings
# -----------------------------
INPUT_DIR = "cron_inputs"
ARCHIVE_DIR = "cron_archive"
OUTPUT_DIR = "outputs"
ERROR_DIR = "error_logs"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)


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
            # Create Excel FIRST
            # ------------------
            excel_path = os.path.join(OUTPUT_DIR, f"{run_id}.xlsx")
            create_excel_from_result(result, excel_path)

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
                "ended_at": datetime.utcnow(),
                "confidence": result.get("confidence"),
            }

            safe_payload = stringify_keys(mongo_payload)

            runs_collection.update_one(
                {"run_id": run_id},
                {"$set": safe_payload},
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

            print(f"[CRON] FAILED {filename}")
            print(tb)


if __name__ == "__main__":
    run_batch()
