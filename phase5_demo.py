"""
Standalone Phase 5 Demo - Area Calculation & Takeoff
=====================================================

Complete Phase 5 demonstration with hardcoded configurations.
Just upload a PDF and select page number - everything else is automatic!

Perfect for demonstrating the full calculation workflow to stakeholders.
"""

import gradio as gr
import json
import sys
import os
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import fitz  # PyMuPDF
import cv2
import io
import re

# Add pipeline to path to allow importing sam3_segmentation
sys.path.insert(0, str(Path(__file__).parent))

# ==========================================
# HARDCODED CONFIGURATIONS
# ==========================================

# These would normally come from Phase 2 (Scale Detection)
# Format: {page_number: {view_label: pixels_per_foot}}
HARDCODED_SCALE_DATA = {
    1: {"Main Elevation": 48.5, "North Elevation": 48.5, "South Elevation": 48.5},
    2: {"Main Elevation": 48.5, "East Elevation": 48.5, "West Elevation": 48.5},
    3: {"Main Elevation": 48.5},
    4: {"Main Elevation": 48.5},
    5: {"Main Elevation": 48.5},
    6: {"Main Elevation": 48.5},
    7: {"Main Elevation": 48.5},
    8: {"Main Elevation": 48.5},
    9: {"Main Elevation": 48.5},
    10: {"Main Elevation": 48.5},
}

# These would normally come from Phase 3 (Tag Detection)
# Format: {page_number: {view_label: {tag: count}}}
HARDCODED_SURVEY_DATA = {
    1: {
        "Main Elevation": {"A": 12, "B": 8, "W1": 15, "W2": 10, "D1": 5},
        "North Elevation": {"A": 10, "W1": 12, "W2": 8},
    },
    2: {
        "Main Elevation": {"A": 8, "B": 6, "W1": 10, "W2": 6, "D1": 3},
        "East Elevation": {"A": 6, "W1": 8},
    },
    3: {"Main Elevation": {"A": 5, "W1": 8, "W2": 4}},
    4: {"Main Elevation": {"A": 4, "W1": 6, "W2": 3}},
    5: {"Main Elevation": {"A": 6, "W1": 10}},
    6: {"Main Elevation": {"A": 3, "W1": 5}},
    7: {"Main Elevation": {"A": 7, "W1": 9, "W2": 5}},
    8: {"Main Elevation": {"A": 4, "W1": 6}},
    9: {"Main Elevation": {"A": 5, "W1": 7, "W2": 4}},
    10: {"Main Elevation": {"A": 6, "W1": 8, "W2": 5}},
}

# Project specifications - window and door dimensions
HARDCODED_PROJECT_SPECS = {
    "windows": {
        "W1": {"width": 3.0, "height": 5.0, "category": "Window"},
        "W2": {"width": 4.0, "height": 6.0, "category": "Window"},
        "W3": {"width": 5.0, "height": 7.0, "category": "Window"},
    },
    "doors": {
        "A": {"width": 3.0, "height": 7.0, "category": "Door"},
        "B": {"width": 6.0, "height": 8.0, "category": "Overhead Door"},
        "D1": {"width": 3.5, "height": 8.0, "category": "Door"},
        "D2": {"width": 4.0, "height": 8.0, "category": "Door"},
    }
}

# Placeholder for Ollama model name, if fallback is needed
MODEL_NAME = "qwen3-vl:30b-a3b-instruct"


# ==========================================
# 1. VIEW DETECTOR (From Phase 3)
# ==========================================
def detect_drawing_views(page_image, prompt="elevation", use_sam3=True):
    """
    Detect distinct drawing views on an architectural sheet using SAM3 (Roboflow).
    
    Args:
        page_image: PIL Image of the page
        prompt: Class name to detect (e.g. "elevation", "building")
        use_sam3: Whether to use SAM3 (forced True)
    
    Returns:
        List of dicts with 'label' and 'box_1000' keys
    """
    img_array = np.array(page_image)
    
    try:
        from pipeline.sam3_segmentation import segment_building_automatic
        
        print(f"[SAM3] Detecting '{prompt}' regions...")
        masks, masks_info = segment_building_automatic(img_array, prompt=prompt, min_area_ratio=0.005)
        
        if masks_info and len(masks_info) > 0:
            views = []
            img_h, img_w = img_array.shape[:2]
            
            # Sort by area
            sorted_masks = sorted(masks_info, key=lambda x: x['area'], reverse=True)
            
            for idx, mask_info in enumerate(sorted_masks[:10]): 
                bbox = mask_info['bbox']
                x, y, w, h = bbox
                
                # Convert to normalized coordinates
                x0_norm = int((x / img_w) * 1000)
                y0_norm = int((y / img_h) * 1000)
                x1_norm = int(((x + w) / img_w) * 1000)
                y1_norm = int(((y + h) / img_h) * 1000)
                
                label = mask_info.get('class_name', prompt.capitalize())
                area_ratio = mask_info['area'] / (img_h * img_w)
                
                views.append({
                    "label": f"{label} {idx+1}" if idx > 0 else label,
                    "box_1000": [x0_norm, y0_norm, x1_norm, y1_norm],
                    "confidence": float(mask_info.get('stability_score', 1.0)),
                    "area_ratio": float(area_ratio),
                    "method": "SAM3"
                })
            
            return views
            
    except Exception as e:
        print(f"[SAM3] Error: {e}")
    
    return []


# ==========================================
# 2. VECTOR ENGINE (From Phase 5)
# ==========================================
def get_vector_mask_area(page, crop_rect, px_per_ft_base):
    # Render at 150 DPI (Zoom 2.0) for speed/precision balance
    zoom = 2.0 
    mat = fitz.Matrix(zoom, zoom)
    
    try:
        pix = page.get_pixmap(matrix=mat, clip=crop_rect)
    except: return 0.0, None

    img_data = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img_data.reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Binarize & Morph
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return 0.0, Image.fromarray(gray)

    # Smart Filter
    img_h, img_w = gray.shape
    total_area = img_h * img_w
    valid_contours = []
    
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < (total_area * 0.005): continue # Too small (noise)
        if area_px > (total_area * 0.95): continue # Too big (border)
        
        _, _, cw, ch = cv2.boundingRect(cnt)
        if ch > 0:
            aspect = cw/ch
            if aspect > 40 or aspect < 0.02: continue # Lines
            
        valid_contours.append(cnt)
    
    if not valid_contours: return 0.0, Image.fromarray(gray)

    # Pick largest valid shape
    target_cnt = max(valid_contours, key=cv2.contourArea)
    area_px_zoom = cv2.contourArea(target_cnt)
    
    # Calculate Square Footage
    # Real Scale = Base Scale * Zoom Factor
    real_scale = px_per_ft_base * zoom
    gross_sqft = area_px_zoom / (real_scale ** 2) if real_scale > 0 else 0
    
    # Draw Green Overlay
    vis_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(vis_img, [target_cnt], -1, (0, 255, 0), -1)
    alpha = 0.4
    overlay = cv2.addWeighted(vis_img, alpha, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), 1 - alpha, 0)
    
    return gross_sqft, Image.fromarray(overlay)


# ==========================================
# 3. DEDUCTION CALCULATOR (From Phase 5)
# ==========================================
def get_deduction_line_items(view_counts, project_specs, page_num, view_label):
    line_items = []
    total_deduction_area = 0.0
    
    # Flatten library
    all_types = {}
    all_types.update(project_specs.get('windows', {}))
    all_types.update(project_specs.get('doors', {}))

    for tag, count in view_counts.items():
        clean_tag = str(tag).upper().replace("TYPE", "").strip()
        spec = None
        
        if clean_tag in all_types:
            spec = all_types[clean_tag]
        else:
            # Fuzzy match
            for key, val in all_types.items():
                if clean_tag == key.replace("-", "") or key == clean_tag.replace("-", ""):
                    spec = val
                    break
        
        if spec:
            try:
                w = float(spec.get('width', 0))
                h = float(spec.get('height', 0))
                cat = spec.get('category', 'Opening').title()
                
                if w > 0 and h > 0:
                    unit_area = w * h
                    total_item_area = unit_area * count
                    total_deduction_area += total_item_area
                    
                    line_items.append({
                        "Page": page_num,
                        "View": view_label,
                        "Category": "Deduction",
                        "Description": f"{cat} {clean_tag}",
                        "Dimensions": f"{w:.2f}' x {h:.2f}'",
                        "Count": count,
                        "Unit_SF": round(unit_area, 2),
                        "Total_SF": round(-total_item_area, 2)
                    })
            except: pass
            
    return line_items, total_deduction_area


# ==========================================
# 4. EXECUTION ENGINE (From Phase 5)
# ==========================================
def execute(pdf_path, survey_data, scale_data, project_specs, target_prompt="elevation"):
    print(f"--- [SAM3] Extracting '{target_prompt}' ---")
    doc = fitz.open(pdf_path)
    
    all_line_items = []
    debug_gallery = []
    grand_total_net = 0.0
    
    # Iterate through pages
    for page_num, views_config in survey_data.items():
        if page_num not in scale_data: continue
        print(f"  > Processing P{page_num}...")
        
        page = doc[page_num - 1]
        
        # Low-res rendering for speed (1.0x = 72 DPI)
        pix_low = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
        page_img_pil = Image.open(io.BytesIO(pix_low.tobytes("jpg")))
        detected_views = detect_drawing_views(page_img_pil, prompt=target_prompt)
        
        if not detected_views:
            print(f"  ⚠️ No '{target_prompt}' found on page {page_num}")
            continue
        page_img_pil = Image.open(io.BytesIO(pix_low.tobytes("png")))
        detected_views = detect_drawing_views(page_img_pil)
        
        # VISUALIZE SAM3 RESULTS
        vis_sam3 = page_img_pil.copy()
        draw_sam3 = ImageDraw.Draw(vis_sam3)
        img_w, img_h = vis_sam3.size
        
        found_classes = []
        methods = []
        for view in detected_views:
            box = view.get("box_1000", [0,0,1000,1000])
            label = view.get("label", "?")
            conf = view.get("confidence", 0.0)
            meth = view.get("method", "Unknown")
            found_classes.append(label)
            methods.append(meth)
            
            x0 = (box[0]/1000) * img_w
            y0 = (box[1]/1000) * img_h
            x1 = (box[2]/1000) * img_w
            y1 = (box[3]/1000) * img_h
            
            # Draw box and label
            color = "blue" if meth == "SAM3" else "red"
            draw_sam3.rectangle([x0, y0, x1, y1], outline=color, width=4)
            draw_sam3.text((x0+10, y0+10), f"[{meth}] {label}", fill=color)
            
        best_method = "/".join(sorted(list(set(methods)))) if methods else "None"
        class_summary = ", ".join(set(found_classes)) if found_classes else "None found"
        debug_gallery.append((vis_sam3, f"Extracted via {best_method}: {class_summary}"))
        
        for view_meta in detected_views:
            label = view_meta.get("label", "Main")
            box_1000 = view_meta.get("box_1000")
            
            # Match View Label to Scale Data
            view_scale = 0.0
            page_scales = scale_data[page_num]
            
            if label in page_scales:
                view_scale = page_scales[label]
            else:
                # Fallback: Use any available scale for this page
                if page_scales:
                    view_scale = list(page_scales.values())[0]
            
            if view_scale < 1.0: continue

            # Crop Rect
            w, h = page.rect.width, page.rect.height
            crop_rect = fitz.Rect(
                (box_1000[0]/1000)*w, (box_1000[1]/1000)*h,
                (box_1000[2]/1000)*w, (box_1000[3]/1000)*h
            )
            
            # 1. Gross Calc
            gross_sqft, debug_img = get_vector_mask_area(page, crop_rect, view_scale)
            if gross_sqft < 50: continue

            # 2. Deduction Calc
            view_tags = views.get(label, {})
            # If label mismatch, try to merge all counts on page?
            # Safe bet: If survey has specific view keys, use them. If generic, use generic.
            if not view_tags and len(views) == 1:
                view_tags = list(views.values())[0]

            deduct_items, deduct_total = get_deduction_line_items(view_tags, project_specs, page_num, label)
            
            # 3. Sanity Clamp (Max 40% deduction)
            clamped_deduct = deduct_total
            note = ""
            if deduct_total > (gross_sqft * 0.40):
                clamped_deduct = gross_sqft * 0.40
                note = " (Auto-Clamped)"
                ratio = clamped_deduct / deduct_total if deduct_total > 0 else 0
                for item in deduct_items:
                    item['Total_SF'] *= ratio
                    item['Description'] += " [Clamped]"

            net_sqft = gross_sqft - clamped_deduct
            grand_total_net += net_sqft
            
            # Add to Report
            all_line_items.append({
                "Page": page_num, "View": label, "Category": "EIFS Wall",
                "Description": "Gross Facade Area (Vector)", "Dimensions": "-",
                "Count": 1, "Unit_SF": round(gross_sqft, 2), "Total_SF": round(gross_sqft, 2)
            })
            all_line_items.extend(deduct_items)
            
            debug_gallery.append((debug_img, f"P{page_num} {label}\\nNet: {net_sqft:.0f} sf{note}"))
            print(f"    - {label}: Gross {gross_sqft:.0f} - Ded {clamped_deduct:.0f} = Net {net_sqft:.0f}")

    return all_line_items, grand_total_net, debug_gallery


# ==========================================
# SAM3 CHECK FUNCTION
# ==========================================

def check_sam3_configuration():
    """Check Roboflow Docker status"""
    lines = []
    
    url = os.getenv("ROBOFLOW_API_URL", "http://localhost:9001")
    lines.append(f"🌐 Roboflow Server: {url}")
    
    # Connection Check
    try:
        import requests
        try:
            requests.get(url, timeout=2) 
            lines.append("   ✅ Connection Successful")
        except:
             lines.append("   ❌ Connection Failed")
             lines.append("   ⚠️ Ensure Docker container is running on port 9001")
    except:
        pass
        
    return "\n".join(lines)


# ==========================================
# PROCESSING FUNCTION
# ==========================================

def process_phase5_simple(pdf_file, page_number, target_prompt):
    """
    Process SAM3 extraction with custom prompt
    """
    try:
        if pdf_file is None:
            return None, "❌ **Error:** Please upload a PDF file", []
        
        # Check PDF
        doc = fitz.open(pdf_file.name)
        total_pages = len(doc)
        doc.close()
        
        page_num = int(page_number)
        if page_num < 1 or page_num > total_pages:
            return None, f"❌ **Error:** Page {page_num} invalid. PDF has {total_pages} pages.", []
        
        # Prepare data for calculations
        survey_data = {page_num: HARDCODED_SURVEY_DATA.get(page_num, {"Main Elevation": {}})}
        scale_data = {page_num: HARDCODED_SCALE_DATA.get(page_num, {"Main Elevation": 48.5})}
        
        # Execute extraction
        line_items, grand_total, debug_gallery = execute(
            pdf_path=pdf_file.name,
            survey_data=survey_data,
            scale_data=scale_data,
            project_specs=HARDCODED_PROJECT_SPECS,
            target_prompt=target_prompt
        )
        
        print(f"\n{'='*60}")
        print(f"✅ COMPLETED")
        print(f"📊 {len(line_items)} line items generated")
        print(f"📐 Grand Total: {grand_total:.2f} sqft")
        print(f"{'='*60}\n")
        
        # Convert to DataFrame
        if line_items:
            df = pd.DataFrame(line_items)
        else:
            df = pd.DataFrame(columns=["Page", "View", "Category", "Description", "Dimensions", "Count", "Unit_SF", "Total_SF"])
        
        # Create summary
        summary = f"## ✅ Phase 5 Complete\n\n"
        summary += f"**Page:** {page_num} of {total_pages}\n\n"
        summary += f"**Grand Total Net Area:** {grand_total:,.2f} sqft\n\n"
        summary += f"**Line Items:** {len(line_items)}\n\n"
        summary += f"**Views Processed:** {len(debug_gallery)}\n\n"
        
        if grand_total > 0:
            summary += "---\n\n"
            summary += "### 📋 Configuration Used\n\n"
            summary += f"**Scale:** {list(scale_data[page_num].values())[0]:.1f} pixels/foot\n\n"
            summary += f"**Tags Detected:** {sum(sum(v.values()) for v in survey_data[page_num].values())} total\n"
        
        # Add all PDF pages + results to gallery
        # Add only the PROCESSED page to gallery
        doc = fitz.open(pdf_file.name)
        gallery = []
        
        # 1. Show Original Input Page
        page = doc[page_num - 1] # 0-indexed
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        gallery.append((img, f"Page {page_num} (Original Input)"))
        
        doc.close()
        
        # 2. Add processing results (SAM3 Detections & Final Calculations)
        for img, caption in debug_gallery:
            # Customize label based on content
            if "SAM3" in caption:
                 label = f"SAM3 Segmentation: Found {caption.split(' ')[0]} Objects" 
                 # Or just pass the caption if it's already descriptive
                 gallery.append((img, caption))
            else:
                 gallery.append((img, f"Calculation: {caption}"))
        
        return df, summary, gallery
        
    except Exception as e:
        import traceback
        error = f"❌ **Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return None, error, []


# ==========================================
# GRADIO INTERFACE
# ==========================================

def create_interface():
    """Create the Gradio interface"""
    
    initial_sam3_status = check_sam3_configuration()
    
    with gr.Blocks(title="Phase 5 Demo - Area Calculation") as demo:
        gr.Markdown("""
        # 🏗️ Phase 5 Demo - Detailed Area Calculation
        
        **Complete Phase 5 workflow demonstration with hardcoded configurations.**
        
        Just upload your PDF and select which page to analyze - all other inputs are pre-configured!
        
        ### What This Does:
        1. Uses SAM3 to detect building regions on the page
        2. Calculates gross facade area in square feet
        3. Applies deductions for detected windows/doors
        4. Generates detailed line item breakdown
        5. Shows visual debug output with green overlays
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Inputs")
                
                pdf_input = gr.File(
                    label="📄 Upload PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                page_input = gr.Number(
                    label="📖 Page Number",
                    value=1,
                    minimum=1,
                    precision=0,
                    info="Which page to analyze"
                )
                
                prompt_input = gr.Textbox(
                    label="🔍 Class / Prompt",
                    value="elevation",
                    placeholder="e.g. building, elevation, scale, window"
                )
                
                run_btn = gr.Button("🚀 Run SAM3 Extraction", variant="primary", size="lg")
                
                with gr.Accordion("⚙️ Roboflow Status", open=False):
                    sam3_status = gr.Textbox(
                        value=initial_sam3_status,
                        label="Configuration",
                        lines=10,
                        interactive=False
                    )
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
                
                with gr.Accordion("📋 Hardcoded Configurations", open=False):
                    gr.Markdown(f"""
                    **Scale Data (pixels/foot):**
                    ```json
                    {json.dumps(HARDCODED_SCALE_DATA, indent=2)}
                    ```
                    
                    **Project Specs (window/door sizes):**
                    ```json
                    {json.dumps(HARDCODED_PROJECT_SPECS, indent=2)}
                    ```
                    
                    **Survey Data:** See details for each page in preprocessing
                    """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 📤 Results")
                
                summary_output = gr.Markdown("Ready to process...")
                
                with gr.Tabs():
                    with gr.Tab("📊 Line Items"):
                        gr.Markdown("Detailed breakdown of all calculations")
                        line_items_table = gr.Dataframe(
                            headers=["Page", "View", "Category", "Description", "Dimensions", "Count", "Unit_SF", "Total_SF"],
                            interactive=False,
                            wrap=True
                        )
                    
                    with gr.Tab("🖼️ Visual Debug"):
                        gr.Markdown("""
                        **PDF Preview:** All pages from your PDF
                        - ✅ PROCESSED = Analyzed by Phase 5
                        - 📄 Preview = Not processed
                        
                        **RESULT:** Processing output with green overlay showing detected building area
                        """)
                        gallery_output = gr.Gallery(
                            label="Pages & Results",
                            columns=3,
                            height="auto"
                        )
        
        # Wire up
        refresh_btn.click(
            fn=check_sam3_configuration,
            outputs=[sam3_status]
        )
        
        run_btn.click(
            fn=process_phase5_simple,
            inputs=[pdf_input, page_input, prompt_input],
            outputs=[line_items_table, summary_output, gallery_output]
        )
        
        gr.Markdown("""
        ---
        ### 📝 Understanding the Results
        
        **Line Items Table:**
        - **Gross Facade Area** - Total building area detected by SAM3
        - **Deductions** - Windows and doors subtracted from gross area
        - **Total_SF** - Final square footage for each item
        
        **Visual Debug:**
        - Green overlay shows the detected building mask
        
        ### 🔧 Notes
        - Scale, survey data, and project specs are hardcoded for demo
        - In production, these come from Phases 2 and 3
        - SAM3 provides automatic building region detection
        - NO results if SAM3 not configured
        """)
    
    return demo


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("="*60)
    print("PHASE 5 DEMO - AREA CALCULATION")
    print("="*60)
    
    sam3_status = check_sam3_configuration()
    print("\nSAM3 Status:")
    print(sam3_status)
    print("\n" + "="*60 + "\n")
    
    demo = create_interface()
    # Launch with public access enabled for sharing
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
