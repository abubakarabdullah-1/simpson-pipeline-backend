import fitz  # PyMuPDF
import ollama
import io
import re
import json
import logging
from PIL import Image, ImageDraw

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "qwen3-vl:30b-a3b-instruct"

# ==========================================
# 1. VISUAL DEBUGGER
# ==========================================
def generate_debug_thumbnail(page, category, reason):
    try:
        # Increased resolution from 0.3 to 1.5 for better quality
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)) 
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        draw = ImageDraw.Draw(img)
        
        is_actionable = category in ["Exterior_Elevation", "Schedule", "Type_Definition", "Floor_Plan"]
        color = (0, 255, 0) if is_actionable else (255, 0, 0)
        
        # Only draw border, no text (text is added by debug_pdf_collector)
        draw.rectangle([0, 0, img.width-1, img.height-1], outline=color, width=5)
        return img
    except: return None

# ==========================================
# 2. FORENSICS (FIXED CRASH)
# ==========================================
def get_page_stats(page):
    img_area = 0.0
    try:
        # FIX: full=True is required for get_image_bbox to work reliably
        images = page.get_images(full=True)
        for img in images:
            try:
                bbox = page.get_image_bbox(img)
                img_area += bbox.get_area()
            except Exception:
                # If an image is a mask or corrupted, skip it; don't crash
                continue
    except Exception:
        pass
        
    page_area = page.rect.width * page.rect.height
    raster_pct = (img_area / page_area) * 100 if page_area > 0 else 0
    
    # Vector Count (Line Drawings have high count)
    vector_count = len(page.get_drawings())
    
    return {"raster": raster_pct, "vector_count": vector_count}

def analyze_title_block(page):
    # Scan bottom right 15% of page for Sheet ID / Title
    r = page.rect
    clip = fitz.Rect(r.width*0.6, r.height*0.85, r.width, r.height)
    text = page.get_text("text", clip=clip).upper()
    return text

# ==========================================
# 3. RULE-BASED CLASSIFIER
# ==========================================
def classify_by_rules(text, stats):
    # 1. IMMEDIATE REJECTS (The "Cover Sheet" & "Assembly" Filter)
    bad_keywords = [
        "SHEET INDEX", "DRAWING LIST", "COVER SHEET", 
        "SYMBOLS", "ABBREVIATIONS", "LOCATION MAP", 
        "PERSPECTIVE", "RENDERING", "3D VIEW",
        "TRASH ENCLOSURE", "GAZEBO", "FENCING", "MONUMENT SIGN"
    ]
    if any(k in text for k in bad_keywords):
        return "Irrelevant", "Keyword Blocked"

    # 2. SECTIONS REJECTS
    if "SECTION" in text and "ELEVATION" not in text:
        return "Irrelevant", "Section View"
    if "ENLARGED" in text:
        return "Irrelevant", "Enlarged View"

    # 3. POSITIVE MATCHES
    if "EXTERIOR ELEVATION" in text or "BUILDING ELEVATION" in text:
        # RASTER GUARD: If it says "Elevation" but is >40% image, it's a render.
        if stats['raster'] > 40.0:
            return "Irrelevant", f"High Raster ({stats['raster']:.1f}%) - Likely Render"
        return "Exterior_Elevation", "Title Block Match"
    
    if "SCHEDULE" in text:
        if "DOOR" in text or "WINDOW" in text or "LOUVER" in text or "FINISH" in text:
            return "Schedule", "Title Block Match"
            
    if "TYPES" in text or "DETAILS" in text:
        if "WINDOW" in text or "DOOR" in text:
            return "Type_Definition", "Title Block Match"

    if "FLOOR PLAN" in text:
        return "Floor_Plan", "Title Block Match"

    return None, None

# ==========================================
# 4. VLM CLASSIFIER (Fallback)
# ==========================================
def classify_by_vlm(page_image):
    byte_arr = io.BytesIO()
    page_image.save(byte_arr, format='PNG')
    
    prompt = """
    Classify this architectural sheet.
    OPTIONS:
    1. "Exterior_Elevation": 2D Orthographic line drawing of facade. (NOT 3D Renders).
    2. "Schedule": Spreadsheet/Table of Door/Window data.
    3. "Type_Definition": Isolated drawings of specific windows/doors.
    4. "Floor_Plan": Top down view.
    5. "Irrelevant": Cover sheets, Indexes, Sections, Details, 3D Perspectives, Trash Enclosures.
    
    OUTPUT JSON: {"class": "Category", "reason": "..."}
    """
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt, 'images': [byte_arr.getvalue()]}])
        match = re.search(r"\{.*\}", response['message']['content'], re.DOTALL)
        if match: return json.loads(match.group(0))
    except: pass
    return {"class": "Irrelevant", "reason": "VLM Failed"}

# ==========================================
# 5. EXECUTE
# ==========================================
def execute(pdf_path, max_pages=50):
    print("--- [Phase 1] Strict Classification ---")
    doc = fitz.open(pdf_path)
    results = []
    
    scan_limit = len(doc)
    
    for i in range(scan_limit):
        page = doc[i]
        # Call Safe Stats
        stats = get_page_stats(page)
        tb_text = analyze_title_block(page)
        
        # Rule Check
        cat, reason = classify_by_rules(tb_text, stats)
        
        # VLM Fallback
        if not cat:
            pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            vlm_res = classify_by_vlm(img)
            cat = vlm_res.get("class", "Irrelevant")
            reason = vlm_res.get("reason", "VLM")
            
            # Double check VLM result against Raster Guard
            if cat == "Exterior_Elevation" and stats['raster'] > 40.0:
                cat = "Irrelevant"
                reason = "VLM Overridden by Raster Guard"

        print(f"  P{i+1}: [{cat}] ({reason})")
        thumb = generate_debug_thumbnail(page, cat, reason)
        
        results.append({
            "page": i+1,
            "type": cat,
            "thumb": thumb
        })
        
    return results