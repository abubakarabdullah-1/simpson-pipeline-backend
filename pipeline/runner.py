from pipeline import (
    phase1_v3,
    phase2_v3,
    phase3_v4,
    phase4_v3,
    phase5_v2,
)


# ==========================================
# CONFIDENCE CALCULATOR........
# ==========================================
def compute_confidence(
    elevation_pages,
    project_specs,
    survey_data,
    scale_data,
    line_items,
    grand_total,
):
    score = 0
    max_score = 100
    
    # Total potential elevation pages (denominator)
    total_pages = len(elevation_pages) if elevation_pages else 0

    # --- Phase 1: Found elevations (20 pts)
    # Binary gate: If we found any elevations, we get these points.
    if total_pages > 0:
        score += 20

    # --- Phase 2: Specs extracted (20 pts)
    # Split into Windows (10) and Doors (10)
    windows_found = len(project_specs.get("windows", {})) > 0
    doors_found = len(project_specs.get("doors", {})) > 0
    
    if windows_found:
        score += 10
    if doors_found:
        score += 10

    # --- Phase 3: Survey success (25 pts)
    # Proportional: (Pages with >0 survey hits / Total Pages) * 25
    if total_pages > 0:
        pages_with_survey = 0
        for p_num, p_data in survey_data.items():
            # p_data is a dict of view_label -> counts
            # We check if any view found any items
            hits = sum(len(v) for v in p_data.values())
            if hits > 0:
                pages_with_survey += 1
        
        survey_ratio = pages_with_survey / total_pages
        score += (survey_ratio * 25)

    # --- Phase 4: Scale calibrated (25 pts)
    # Proportional: (Pages with scale / Total Pages) * 25
    if total_pages > 0:
        pages_with_scale = 0
        for p_num, p_data in scale_data.items():
            # p_data is dict of view_label -> scale_val
            if len(p_data) > 0:
                pages_with_scale += 1
                
        scale_ratio = pages_with_scale / total_pages
        score += (scale_ratio * 25)

    # --- Phase 5: Final quantities produced (10 pts)
    # Binary bonus for final output
    if grand_total and grand_total > 0:
        score += 10

    return round(min(score, max_score), 2)


# ==========================================
# PIPELINE RUNNER
# ==========================================
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

    # -------------------------
    # CONFIDENCE
    # -------------------------
    confidence = compute_confidence(
        elevation_pages,
        project_specs,
        survey_data,
        scale_data,
        line_items,
        grand_total,
    )

    return {
        "project_specs": project_specs,
        "survey_data": survey_data,
        "scale_data": scale_data,
        "line_items": line_items,
        "grand_total": grand_total,
        "logs": logs,
        "confidence": confidence,
    }
