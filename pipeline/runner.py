from pipeline import (
    phase1_v3,
    phase2_v3,
    phase3_v4,
    phase4_v3,
    phase5_v2,
)

from pipeline.debug_pdf_collector import collect_and_write_debug_pdf
from pipeline.validator import validate_step
from pipeline.log_collector import LogCollector

import os
import threading
from datetime import datetime
# MongoDB Connection for heartbeat updates
# Avoid initializing here to prevent double connection / DNS timeout on import
runs_collection = None

# Shared cancellation event — set by app.py to stop all pipelines
cancel_event = threading.Event()

_HEARTBEAT_INTERVAL = 30  # seconds between background heartbeats


def is_cancelled():
    """Check if the pipeline run has been cancelled."""
    return cancel_event.is_set()


def update_heartbeat(run_id: str, collection=None):
    """Update last_updated timestamp to show pipeline is alive"""
    target_collection = collection or runs_collection
    if target_collection is not None and run_id:
        try:
            target_collection.update_one(
                {"run_id": run_id},
                {"$set": {"last_updated": datetime.utcnow()}}
            )
        except Exception:
            pass  # Fail silently on heartbeat to avoid crashing pipeline


def update_progress(run_id: str, phase: str, progress: int):
    """
    Print pipeline progress to console/logs
    
    Args:
        run_id: Pipeline run ID
        phase: Current phase name
        progress: Progress percentage (0-100)
    """
    print(f"📊 [{run_id}] Progress: {progress}% - {phase}")



# ==========================================
# PIPELINE RUNNER — FINAL VERSION
# ==========================================


class _HeartbeatThread:
    """Background thread that sends heartbeats at regular intervals."""
    def __init__(self, run_id, interval=_HEARTBEAT_INTERVAL):
        self.run_id = run_id
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            update_heartbeat(self.run_id)
            self._stop.wait(self.interval)

    def start(self):
        if self.run_id:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def run_pipeline(pdf_path: str, run_id: str = None):

    logs = []
    
    # Initialize log collector
    log_collector = LogCollector()

    # Start continuous background heartbeat so no phase can starve the monitor
    hb = _HeartbeatThread(run_id)
    hb.start()

    def log(msg: str):
        print(msg)
        logs.append(msg)

    log("Starting pipeline")

    try:
        # -------------------------
        # PHASE 1
        # -------------------------
        update_heartbeat(run_id)
        with log_collector.capture_phase("phase1"):
            actionable_pages = phase1_v3.execute(pdf_path)

        elevation_pages = [
            p["page"]
            for p in actionable_pages
            if p["type"] == "Exterior_Elevation"
        ]

        # Phase 1 thumbnails are just re-renders of original pages with
        # a colored border — not real pipeline output.  Original pages are
        # already saved as page_N.pdf, so skip phase1 debug images.
        phase1_debug = []

        update_progress(run_id, "Phase 1: PDF Processing Complete", 20)

        # -------------------------
        # PHASE 2
        # -------------------------
        update_heartbeat(run_id)
        with log_collector.capture_phase("phase2"):
            project_specs, phase2_debug = phase2_v3.execute(
                pdf_path,
                actionable_pages,
            )

        update_progress(run_id, "Phase 2: Element Detection Complete", 40)

        # -------------------------
        # PHASE 3
        # -------------------------
        update_heartbeat(run_id)
        with log_collector.capture_phase("phase3"):
            survey_data, phase3_debug = phase3_v4.execute(
                pdf_path,
                elevation_pages,
                project_specs,
            )

        update_progress(run_id, "Phase 3: Dimension Extraction Complete", 60)

        # -------------------------
        # PHASE 4
        # -------------------------
        update_heartbeat(run_id)
        with log_collector.capture_phase("phase4"):
            scale_data, phase4_debug = phase4_v3.execute(
                pdf_path,
                elevation_pages,
            )

        update_progress(run_id, "Phase 4: Scale Analysis Complete", 80)

        # -------------------------
        # PHASE 5
        # -------------------------
        update_heartbeat(run_id)
        with log_collector.capture_phase("phase5"):
            line_items, grand_total, phase5_debug = phase5_v2.execute(
                pdf_path,
                survey_data,
                scale_data,
                project_specs,
            )

        update_progress(run_id, "Phase 5: Output Generation Complete", 100)

        # -------------------------
        # VALIDATOR CONFIDENCE
        # -------------------------

        validator_scores = []

        for phase_debug in [
            phase3_debug,
            phase4_debug,
            phase5_debug,
        ]:

            if not phase_debug:
                continue

            for img, label in phase_debug:

                try:
                    res = validate_step(
                        img,
                        candidate_data={"label": label},
                        phase_context="Pipeline Debug Artifact",
                    )

                    score = float(res.get("confidence_score", 0.0))
                    validator_scores.append(score)

                except Exception:
                    continue

        if validator_scores:
            confidence = round(sum(validator_scores) / len(validator_scores), 3)
        else:
            confidence = 0.0

        # -------------------------
        # DEBUG PDF
        # -------------------------

        # Phase 2 debug images are unmodified re-renders of original
        # pages — no annotations.  Skip them as originals are already saved.
        # Only pass phases that produce genuinely new images:
        #   Phase 3: blue view-boxes + green tag-word boxes
        #   Phase 4: cropped dimension lines with blue highlight
        #   Phase 5: grayscale crop with green contour mask
        debug_pdf = collect_and_write_debug_pdf(
            [
                phase3_debug,
                phase4_debug,
                phase5_debug,
            ],
            output_dir="outputs",
            global_confidence=confidence,
            run_id=run_id,
            pdf_path=pdf_path,
            phase_start_index=3,
        )

        # -------------------------
        # SAVE LOGS TO JSON
        # -------------------------
        log_file_path = None
        if run_id:
            log_file_path = log_collector.save_to_json("logs", run_id)

        return {
            "project_specs": project_specs,
            "survey_data": survey_data,
            "scale_data": scale_data,
            "line_items": line_items,
            "grand_total": grand_total,
            "logs": logs,
            "confidence": confidence,
            "debug_pdf": debug_pdf,
            "log_file": log_file_path,
            "phase_logs": log_collector.get_all_logs()
        }
    finally:
        hb.stop()

