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


# ==========================================
# PIPELINE RUNNER — FINAL VERSION
# ==========================================


def run_pipeline(pdf_path: str, run_id: str = None):

    logs = []
    
    # Initialize log collector
    log_collector = LogCollector()

    def log(msg: str):
        print(msg)
        logs.append(msg)

    log("Starting pipeline")

    # -------------------------
    # PHASE 1
    # -------------------------
    with log_collector.capture_phase("phase1"):
        actionable_pages = phase1_v3.execute(pdf_path)

    elevation_pages = [
        p["page"]
        for p in actionable_pages
        if p["type"] == "Exterior_Elevation"
    ]

    # ---- collect Phase-1 thumbs ----
    phase1_debug = []

    for r in actionable_pages:
        img = r.get("thumb")
        if img:
            phase1_debug.append(
                (img, f"P{r['page']} {r['type']}")
            )

    # -------------------------
    # PHASE 2
    # -------------------------
    with log_collector.capture_phase("phase2"):
        project_specs, phase2_debug = phase2_v3.execute(
            pdf_path,
            actionable_pages,
        )

    # -------------------------
    # PHASE 3
    # -------------------------
    with log_collector.capture_phase("phase3"):
        survey_data, phase3_debug = phase3_v4.execute(
            pdf_path,
            elevation_pages,
            project_specs,
        )

    # -------------------------
    # PHASE 4
    # -------------------------
    with log_collector.capture_phase("phase4"):
        scale_data, phase4_debug = phase4_v3.execute(
            pdf_path,
            elevation_pages,
        )

    # -------------------------
    # PHASE 5
    # -------------------------
    with log_collector.capture_phase("phase5"):
        line_items, grand_total, phase5_debug = phase5_v2.execute(
            pdf_path,
            survey_data,
            scale_data,
            project_specs,
        )

    log("Pipeline finished")

    # -------------------------
    # VALIDATOR CONFIDENCE
    # -------------------------

    validator_scores = []

    for phase_debug in [
        phase1_debug,
        phase2_debug,
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

    debug_pdf = collect_and_write_debug_pdf(
        [
            phase1_debug,
            phase2_debug,
            phase3_debug,
            phase4_debug,
            phase5_debug,
        ],
        output_dir="outputs",
        global_confidence=confidence,
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

