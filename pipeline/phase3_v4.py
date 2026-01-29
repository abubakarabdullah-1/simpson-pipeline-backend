import fitz  # PyMuPDF
import ollama
import io
import json
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw

MODEL_NAME = "qwen3-vl:30b-a3b-instruct"

# ==========================================
# 1. VIEW DETECTOR (SAM3-Enhanced Global Scout)
# ==========================================
def detect_drawing_views(page_image, use_sam3=True):
    """
    Detect distinct drawing views on an architectural sheet using SAM3 or VLM.
    
    Args:
        page_image: PIL Image of the page
        use_sam3: Whether to use SAM3 for view detection (default: True)
    
    Returns:
        List of dicts with 'label' and 'box_1000' keys
    """
    # Convert to numpy array for SAM3
    img_array = np.array(page_image)
    
    # Try SAM3 automatic segmentation first
    if use_sam3:
        try:
            from pipeline.sam3_segmentation import segment_building_automatic
            
            print("[SAM3] Detecting drawing views with SAM3...")
            masks, masks_info = segment_building_automatic(img_array, min_area_ratio=0.01, max_area_ratio=0.90)
            
            if masks_info and len(masks_info) > 0:
                # Convert SAM3 masks to bounding boxes
                views = []
                img_h, img_w = img_array.shape[:2]
                
                # Sort masks by area (largest first) and stability score
                sorted_masks = sorted(
                    masks_info, 
                    key=lambda x: x['area'] * x['stability_score'], 
                    reverse=True
                )
                
                for idx, mask_info in enumerate(sorted_masks[:6]):  # Limit to top 6 views
                    bbox = mask_info['bbox']  # [x, y, width, height]
                    x, y, w, h = bbox
                    
                    # Convert to normalized coordinates [x0, y0, x1, y1] in 0-1000 scale
                    x0_norm = int((x / img_w) * 1000)
                    y0_norm = int((y / img_h) * 1000)
                    x1_norm = int(((x + w) / img_w) * 1000)
                    y1_norm = int(((y + h) / img_h) * 1000)
                    
                    # Generate label based on position
                    area_ratio = mask_info['area'] / (img_h * img_w)
                    
                    # If it's very large, it might be the main view
                    if area_ratio > 0.5:
                        label = "Main Elevation"
                    else:
                        # Position-based naming
                        center_y = y + h/2
                        center_x = x + w/2
                        
                        if center_y < img_h * 0.33:
                            pos_y = "Top"
                        elif center_y < img_h * 0.67:
                            pos_y = "Middle"
                        else:
                            pos_y = "Bottom"
                        
                        if center_x < img_w * 0.33:
                            pos_x = "Left"
                        elif center_x < img_w * 0.67:
                            pos_x = "Center"
                        else:
                            pos_x = "Right"
                        
                        label = f"View {idx+1} ({pos_y} {pos_x})"
                    
                    views.append({
                        "label": label,
                        "box_1000": [x0_norm, y0_norm, x1_norm, y1_norm],
                        "confidence": float(mask_info['stability_score']),
                        "area_ratio": float(area_ratio)
                    })
                
                if views:
                    print(f"[SAM3] Detected {len(views)} views using SAM3")
                    return views
                else:
                    print("[SAM3] No valid views detected, falling back to VLM")
            else:
                print("[SAM3] No masks found, falling back to VLM")
                
        except Exception as e:
            print(f"[SAM3] Error in view detection: {e}, falling back to VLM")
    
    # Fallback to VLM (original method)
    print("[VLM] Detecting views with Vision Language Model...")
    byte_arr = io.BytesIO()
    page_image.save(byte_arr, format='PNG')
    img_bytes = byte_arr.getvalue()

    prompt = """
    Analyze this architectural sheet.
    TASK: Return bounding boxes for distinct drawings (e.g., 'North Elevation', 'South Elevation').
    OUTPUT JSON: {"views": [{"label": "Name", "box_1000": [x0, y0, x1, y1]}]}
    """
    
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt, 'images': [img_bytes]}])
        content = response['message']['content']
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match: 
            views = json.loads(match.group(0)).get("views", [])
            if views:
                print(f"[VLM] Detected {len(views)} views using VLM")
                return views
    except Exception as e:
        print(f"[VLM] Error: {e}")
    
    # Ultimate fallback: full page
    print("[FALLBACK] Using full page as single view")
    return [{"label": "Full Page View", "box_1000": [0,0,1000,1000]}]




# ==========================================
# 2. GEOFENCING ENGINE (The Spatial Filter)
# ==========================================
def generate_building_mask(page, crop_rect, dilation_px=50):
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=crop_rect)
    img_data = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img_data.reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3: gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else: gray = img

    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    if not contours: return mask, gray.shape

    img_area = gray.shape[0] * gray.shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > (img_area * 0.005) and area < (img_area * 0.95):
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    if dilation_px > 0:
        kernel_dil = np.ones((dilation_px, dilation_px), np.uint8)
        mask = cv2.dilate(mask, kernel_dil, iterations=1)

    return mask, (pix.width, pix.height)


def is_point_in_mask(pt, mask, mask_dims, crop_rect):
    rel_x = pt[0] - crop_rect.x0
    rel_y = pt[1] - crop_rect.y0
    scale_x = mask_dims[0] / crop_rect.width
    scale_y = mask_dims[1] / crop_rect.height
    px = int(rel_x * scale_x)
    py = int(rel_y * scale_y)
    if 0 <= px < mask_dims[0] and 0 <= py < mask_dims[1]:
        return mask[py, px] > 0
    return False

# ==========================================
# 3. TAG MATCHING AGENT (VALIDATOR ONLY)
# ==========================================
def agent_verify_tags(tile_img, unique_candidates, known_tags):
    """
    Asks the VLM to confirm which of the UNIQUE strings are valid tags.
    """
    if not unique_candidates: return []
    
    prompt = f"""
    You are a Surveyor.
    
    [INPUT]
    Image: A crop of a building elevation.
    Candidates Found (OCR): {unique_candidates}
    Project Schedule Codes: {known_tags}
    
    [TASK]
    Filter the 'Candidates Found' list. Return only the items that are definitely **Window or Door Tags** visible in the image.
    
    OUTPUT JSON: {{ "matches": ["W1", "A", "102"] }}
    """
    
    byte_arr = io.BytesIO()
    tile_img.save(byte_arr, format='PNG')
    
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt, 'images': [byte_arr.getvalue()]}])
        content = response['message']['content']
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0)).get("matches", [])
    except: pass
    return []

# ==========================================
# 4. PIPELINE ENTRY POINT
# ==========================================
def execute(pdf_path, elevation_pages, project_specs):
    print(f"--- [Phase 3] Geofenced Surveying ---")
    
    known_tags = []
    known_tags.extend(list(project_specs.get("windows", {}).keys()))
    known_tags.extend(list(project_specs.get("doors", {}).keys()))
    known_tags = [t for t in known_tags if len(str(t)) < 10]
    
    doc = fitz.open(pdf_path)
    survey_results = {} 
    debug_artifacts = [] 

    for page_num in elevation_pages:
        print(f"  > Processing Page {page_num}...")
        try:
            page = doc[page_num - 1]
            survey_results[page_num] = {}
            
            pix_full = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            debug_img = Image.open(io.BytesIO(pix_full.tobytes("png")))
            draw = ImageDraw.Draw(debug_img)
            
            def draw_box(rect, color, width=3):
                sx = debug_img.width / page.rect.width
                sy = debug_img.height / page.rect.height
                draw.rectangle([rect[0]*sx, rect[1]*sy, rect[2]*sx, rect[3]*sy], outline=color, width=width)

            pix_low = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
            views = detect_drawing_views(Image.open(io.BytesIO(pix_low.tobytes("png"))))
            
            for view in views:
                label = view.get("label", "View")
                box = view.get("box_1000", [0,0,1000,1000])
                pdf_w, pdf_h = page.rect.width, page.rect.height
                view_rect = fitz.Rect((box[0]/1000)*pdf_w, (box[1]/1000)*pdf_h, (box[2]/1000)*pdf_w, (box[3]/1000)*pdf_h)
                
                mask, mask_dims = generate_building_mask(page, view_rect, dilation_px=40)
                draw_box([view_rect.x0, view_rect.y0, view_rect.x1, view_rect.y1], (0, 0, 255))
                
                # 1. OCR all text inside mask
                words = page.get_text("words")
                valid_vectors = []
                for w in words:
                    center_pt = ((w[0]+w[2])/2, (w[1]+w[3])/2)
                    if view_rect.contains(fitz.Point(center_pt)):
                        if is_point_in_mask(center_pt, mask, mask_dims, view_rect):
                            valid_vectors.append(w)

                if valid_vectors:
                    pix_v = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=view_rect)
                    view_img = Image.open(io.BytesIO(pix_v.tobytes("png")))
                    
                    # 2. Get Unique Strings for Validation
                    candidate_texts = list(set([v[4] for v in valid_vectors]))
                    
                    # 3. VLM Approves Types (e.g., "Yes, W1 is a tag")
                    confirmed_types = agent_verify_tags(view_img, candidate_texts, known_tags)
                    
                    # 4. PYTHON Counts Occurrences in Original List
                    view_counts = {}
                    for v in valid_vectors:
                        text = v[4]
                        # Check if this text vector matches a confirmed type
                        # Fuzzy match: "W1" matches "W1" or "W1."
                        matched_type = None
                        for c_type in confirmed_types:
                            if c_type in text or text in c_type:
                                matched_type = c_type
                                break
                        
                        if matched_type:
                            clean_tag = str(matched_type).strip().upper()
                            view_counts[clean_tag] = view_counts.get(clean_tag, 0) + 1
                            draw_box([v[0], v[1], v[2], v[3]], (0, 255, 0), width=3)
                    
                    if view_counts:
                        survey_results[page_num][label] = view_counts
                        print(f"    - {label}: Found {view_counts}")

            debug_artifacts.append((debug_img, f"P{page_num} Geofence Results"))
            
        except Exception as e:
            print(f"    [Error] P{page_num}: {e}")

    print(f"--- [Phase 3] Complete. ---")
    return survey_results, debug_artifacts