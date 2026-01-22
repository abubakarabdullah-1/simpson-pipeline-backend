from pipeline import (
    phase1_v3,
    phase2_v3,
    phase3_v4,
    phase4_v3,
    phase5_v2,
)


def run_pipeline(pdf_path: str):

    logs = []

    def log(msg: str):
        print(msg)
        logs.append(msg)

    log("Starting pipeline")

    # -------------------------
    # PHASE 1
    # -------------------------
    actionable_pages = phase1_v3.execute(pdf_path)

    elevation_pages = [
        p["page"]
        for p in actionable_pages
        if p["type"] == "Exterior_Elevation"
    ]

    # -------------------------
    # PHASE 2
    # -------------------------
    project_specs, _ = phase2_v3.execute(
        pdf_path,
        actionable_pages,
    )

    # -------------------------
    # PHASE 3
    # -------------------------
    survey_data, _ = phase3_v4.execute(
        pdf_path,
        elevation_pages,
        project_specs,
    )

    # -------------------------
    # PHASE 4
    # -------------------------
    scale_data, _ = phase4_v3.execute(
        pdf_path,
        elevation_pages,
    )

    # -------------------------
    # PHASE 5
    # -------------------------
    line_items, grand_total, _ = phase5_v2.execute(
        pdf_path,
        survey_data,
        scale_data,
        project_specs,
    )

    log("Pipeline finished")

    return {
        "project_specs": project_specs,
        "survey_data": survey_data,
        "scale_data": scale_data,
        "line_items": line_items,
        "grand_total": grand_total,
        "logs": logs,
    }
