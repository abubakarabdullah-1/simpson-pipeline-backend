import gradio as gr
import pandas as pd
import time
import traceback
import os
import json

# Import The Specialist Modules
# Ensure these files are in the same directory:
# phase1.py, phase2.py, phase3.py, phase4.py, phase5_v2.py
import phase1_v3  # Classifier
import phase2_v3  # Spec Extractor
import phase3_v4  # Surveyor
import phase4_v3  # Calibrator
import phase5_v2  # Vector Estimator

def sanitize_keys(data):
    """
    Recursively converts dictionary keys to strings.
    Required because Gradio JSON components crash if keys are Integers (e.g., Page Numbers).
    """
    if isinstance(data, dict):
        return {str(k): sanitize_keys(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_keys(i) for i in data]
    else:
        return data

def run_full_pipeline(pdf_file, progress=gr.Progress()):
    """
    Orchestrates the 5-Phase AI Pipeline with Full Visual Auditing.
    """
    if pdf_file is None:
        return ["No file uploaded."] + [None]*9

    pdf_path = pdf_file.name
    
    # Helper to manage logs
    current_logs = []
    def add_log(msg):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        current_logs.append(entry)
        return "\n".join(current_logs)

    # Initialize State Variables (Prevents UnboundLocalError)
    p1_gallery = []
    project_specs = {}
    p2_debug_crops = []
    survey_data = {}
    p3_heatmaps = []
    scale_data = {}
    p4_proofs = []
    p5_masks = []
    
    # Empty Defaults for UI
    empty_df = pd.DataFrame()
    empty_json = {}
    empty_gal = []

    # Yield State Helper
    # This keeps the UI updated with the latest available data while preserving previous phases
    def yield_current_state(log_txt, det_df=None, sum_df=None):
        return [
            log_txt, 
            p1_gallery or empty_gal, 
            sanitize_keys(project_specs) or empty_json, 
            p2_debug_crops or empty_gal,
            sanitize_keys(survey_data) or empty_json, 
            p3_heatmaps or empty_gal,
            p4_proofs or empty_gal, 
            p5_masks or empty_gal,
            det_df if det_df is not None else empty_df, 
            sum_df if sum_df is not None else empty_df
        ]

    try:
        # ---------------------------------------------------------
        # PHASE 1: CLASSIFICATION
        # ---------------------------------------------------------
        progress(0.1, desc="Phase 1: Scanning PDF...")
        yield yield_current_state(add_log("Phase 1: Identifying Sheet Types..."))
        
        # Execute Phase 1
        actionable_pages = phase1_v3.execute(pdf_path, max_pages=50)
        
        # Extract Visuals (Thumbnails)
        p1_gallery = [(p.get('thumb'), f"P{p['page']}: {p['type']}") for p in actionable_pages]
        
        # Filter Pages
        elevation_pages = [p['page'] for p in actionable_pages if p['type'] == 'Exterior_Elevation']
        definition_pages = [p for p in actionable_pages if p['type'] in ['Schedule', 'Type_Definition', 'Detail', 'Floor_Plan']]
        
        log_msg = f"Phase 1 Complete.\n  - Found {len(elevation_pages)} Elevations\n  - Found {len(definition_pages)} Spec Sources"
        yield yield_current_state(add_log(log_msg))

        if not elevation_pages:
            yield yield_current_state(add_log("STOP: No Elevations found. Cannot proceed."))
            return

        # ---------------------------------------------------------
        # PHASE 2: EXTRACTION
        # ---------------------------------------------------------
        progress(0.3, desc="Phase 2: Mining Specs...")
        yield yield_current_state(add_log("Phase 2: Extracting Windows, Doors, Louvers..."))
        
        # Execute Phase 2
        project_specs, p2_debug_crops = phase2_v3.execute(pdf_path, definition_pages)
        
        w_count = len(project_specs.get("windows", {}))
        d_count = len(project_specs.get("doors", {}))
        
        yield yield_current_state(add_log(f"Phase 2 Done. Cataloged {w_count} Opening Types & {d_count} Door Types."))

        # ---------------------------------------------------------
        # PHASE 3: SURVEYING
        # ---------------------------------------------------------
        progress(0.5, desc="Phase 3: Surveying...")
        yield yield_current_state(add_log("Phase 3: Locating Tags on Elevations..."))
        
        # Execute Phase 3
        survey_data, p3_heatmaps = phase3_v4.execute(pdf_path, elevation_pages, project_specs)
        
        yield yield_current_state(add_log("Phase 3 Complete. Tags located."))

        # ---------------------------------------------------------
        # PHASE 4: CALIBRATION
        # ---------------------------------------------------------
        progress(0.7, desc="Phase 4: Calibrating...")
        yield yield_current_state(add_log("Phase 4: Calibrating Scale per View..."))
        
        # Execute Phase 4
        scale_data, p4_proofs = phase4_v3.execute(pdf_path, elevation_pages)
        
        yield yield_current_state(add_log("Phase 4 Complete. Scale factors calculated."))

        # ---------------------------------------------------------
        # PHASE 5: ESTIMATION
        # ---------------------------------------------------------
        progress(0.9, desc="Phase 5: Estimating...")
        yield yield_current_state(add_log("Phase 5: High-Res Vector Calculation..."))
        
        # Execute Phase 5
        line_items, grand_total, p5_masks = phase5_v2.execute(
            pdf_path, survey_data, scale_data, project_specs
        )
        
        # ---------------------------------------------------------
        # FINAL REPORTING
        # ---------------------------------------------------------
        yield yield_current_state(add_log("Formatting Final Output..."))
        
        # 1. Detailed DataFrame
        if line_items:
            df_detailed = pd.DataFrame(line_items)
            cols = ["Page", "View", "Category", "Description", "Dimensions", "Count", "Unit_SF", "Total_SF"]
            df_detailed = df_detailed[[c for c in cols if c in df_detailed.columns]]
        else:
            df_detailed = pd.DataFrame(columns=["Status"], data=["No items found"])

        # 2. Summary DataFrame
        summary_rows = []
        if not df_detailed.empty and "Total_SF" in df_detailed.columns:
            # Group by Category
            cats = df_detailed.groupby("Category")["Total_SF"].sum().reset_index()
            for _, r in cats.iterrows():
                summary_rows.append({"Item": r["Category"], "Area (SF)": round(r["Total_SF"], 2)})
            
            # Grand Total
            summary_rows.append({"Item": "---", "Area (SF)": "---"})
            summary_rows.append({"Item": ">>> GRAND TOTAL NET EIFS", "Area (SF)": round(grand_total, 2)})
            
        df_summary = pd.DataFrame(summary_rows)

        # Final Success State (Passes the dataframes)
        yield yield_current_state(add_log("Pipeline Finished Successfully."), det_df=df_detailed, sum_df=df_summary)

    except Exception as e:
        err_msg = f"CRITICAL ERROR: {str(e)}\n{traceback.format_exc()}"
        yield yield_current_state(add_log(err_msg))

# --- GRADIO UI LAYOUT ---
with gr.Blocks(title="AI EIFS Estimator V4 - Transparent") as demo:
    gr.Markdown("# 🏗️ AI EIFS Takeoff: The Transparent Edition")
    gr.Markdown("Upload Plans -> Auto-Detect -> Verify Every Step -> Get Net Quantity.")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF Plans", file_types=[".pdf"])
            run_btn = gr.Button("🚀 Run Full Takeoff", variant="primary", size="lg")
            logs_output = gr.Textbox(label="Live Pipeline Logs", lines=25, interactive=False)
            
        with gr.Column(scale=3):
            with gr.Tabs():
                # REPORTING TABS
                with gr.TabItem("📊 Final Estimate"):
                    with gr.Row():
                        summary_table = gr.Dataframe(label="Executive Summary")
                    with gr.Row():
                        detailed_table = gr.Dataframe(label="Detailed Bill of Quantities")
                
                # AUDIT TABS
                with gr.TabItem("1️⃣ Classification"):
                    gr.Markdown("### Phase 1: Sheet Detection (Green=Keep, Red=Skip)")
                    gal_p1 = gr.Gallery(label="Classified Sheets", columns=4, height="auto", object_fit="contain")
                    
                with gr.TabItem("2️⃣ Specs & Types"):
                    gr.Markdown("### Phase 2: What the AI Found (Windows/Doors/Louvers)")
                    with gr.Row():
                        json_p2 = gr.JSON(label="Extracted Type Library")
                        gal_p2 = gr.Gallery(label="AI Vision Crops (What it read)", columns=3, object_fit="contain")
                        
                with gr.TabItem("3️⃣ Survey Counts"):
                    gr.Markdown("### Phase 3: Tag Detection (Green Box = Found Tag)")
                    with gr.Row():
                        json_p3 = gr.JSON(label="Raw Counts per View")
                        gal_p3 = gr.Gallery(label="Elevation Heatmaps", columns=2, object_fit="contain")
                        
                with gr.TabItem("4️⃣ Calibration"):
                    gr.Markdown("### Phase 4: Scale Verification (Blue Line = Ruler)")
                    gal_p4 = gr.Gallery(label="Scale Proofs", columns=2, object_fit="contain")
                    
                with gr.TabItem("5️⃣ Vector Estimation"):
                    gr.Markdown("### Phase 5: Gross Area Detection (Green Mask = Wall)")
                    gal_p5 = gr.Gallery(label="Vector Masks", columns=2, object_fit="contain")

    # Wiring the outputs
    run_btn.click(
        run_full_pipeline, 
        inputs=[pdf_input], 
        outputs=[
            logs_output, 
            gal_p1, 
            json_p2, gal_p2, 
            json_p3, gal_p3, 
            gal_p4, 
            gal_p5, 
            detailed_table, 
            summary_table
        ]
    )

if __name__ == "__main__":
    demo.queue().launch(share=False, server_name="0.0.0.0")