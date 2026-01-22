import fitz  # PyMuPDF
import cv2
import numpy as np
import io
import re
from PIL import Image

# Import the view detector to ensure consistent cropping
# (Ideally, we would pass the view boxes from Phase 3/4 to Phase 5 to avoid re-detection drift)
# For robustness, let's reuse the detector logic here or import it.
try:
    from pipeline.phase3_v4 import detect_drawing_views
except:
    def detect_drawing_views(img): return [{"label": "Full Page", "box_1000": [0,0,1000,1000]}]

# ==========================================
# 1. HELPER: DEDUCTION CALCULATOR
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
# 2. VECTOR ENGINE
# ==========================================
def get_vector_mask_area(page, crop_rect, px_per_ft_base):
    # Render at 300 DPI (Zoom 4.0) for precision
    zoom = 4.0 
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
# 3. PIPELINE ENTRY POINT
# ==========================================
def execute(pdf_path, survey_data, scale_data, project_specs):
    print(f"--- [Phase 5] Detailed Vector Estimation ---")
    doc = fitz.open(pdf_path)
    
    all_line_items = []
    debug_gallery = []
    grand_total_net = 0.0
    
    # Iterate through pages
    for page_num, views in survey_data.items():
        if page_num not in scale_data: continue
        print(f"  > Processing P{page_num}...")
        
        page = doc[page_num - 1]
        
        # Re-detect views to get crop boxes
        pix_low = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
        detected_views = detect_drawing_views(Image.open(io.BytesIO(pix_low.tobytes("png"))))
        
        for view_meta in detected_views:
            label = view_meta.get("label", "Main")
            box_1000 = view_meta.get("box_1000")
            
            # Match View Label to Scale Data
            # Logic: Look for exact match, then fuzzy match in scale_data keys
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
            # Find matching survey counts for this view
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
            
            debug_gallery.append((debug_img, f"P{page_num} {label}\nNet: {net_sqft:.0f} sf{note}"))
            print(f"    - {label}: Gross {gross_sqft:.0f} - Ded {clamped_deduct:.0f} = Net {net_sqft:.0f}")

    return all_line_items, grand_total_net, debug_gallery