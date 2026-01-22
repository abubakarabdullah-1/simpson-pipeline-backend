import fitz  # PyMuPDF
import ollama
import io
import json
import re
from PIL import Image

MODEL_NAME = "qwen3-vl:30b-a3b-instruct"

# ==========================================
# 1. SCHEDULE PARSER (The Table Reader)
# ==========================================
def extract_schedule_data(image):
    """
    Extracts structured data from a Schedule Table.
    PRIORITY: Look for 'Mark', 'Tag', or 'Type' columns.
    """
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    
    prompt = """
    Analyze this Architectural Schedule Table.
    TASK: Extract the Door/Window Marks and their Dimensions.
    
    CRITICAL RULES:
    1. The 'Mark' or 'Tag' column is the ID (e.g., A, B, 101, W1). 
    2. **IGNORE** columns labeled "Frame Type", "Detail", or "Remarks" (e.g., ignore 'TYP-A2', 'HM', 'DTL-1').
    3. Look for Width and Height columns (e.g., 3'-0", 7'-0").
    
    OUTPUT JSON: 
    { "items": [ {"mark": "A", "width_str": "3'-0\"", "height_str": "7'-0\"", "category": "door"} ] }
    """
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt, 'images': [byte_arr.getvalue()]}])
        match = re.search(r"\{.*\}", response['message']['content'], re.DOTALL)
        if match: return json.loads(match.group(0)).get("items", [])
    except: pass
    return []

# ==========================================
# 2. VISUAL LEGEND PARSER (The Drawing Reader)
# ==========================================
def extract_visual_data(image):
    """
    Extracts types defined by drawings (Window Types).
    """
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    
    prompt = """
    Analyze these Component Type drawings.
    TASK: Extract the Type Label and Dimensions for each drawing.
    
    RULES:
    1. Capture the MAIN Label (e.g., "W1", "Type A", "SF-1").
    2. **IGNORE** reference notes like "SIM", "TYP", "OPP", "MIRROR".
    3. Read the dimensions associated with the unit.
    
    OUTPUT JSON:
    { "items": [ {"mark": "W1", "width_str": "4'-0\"", "height_str": "6'-0\"", "category": "window"} ] }
    """
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt, 'images': [byte_arr.getvalue()]}])
        match = re.search(r"\{.*\}", response['message']['content'], re.DOTALL)
        if match: return json.loads(match.group(0)).get("items", [])
    except: pass
    return []

# ==========================================
# 3. HELPER: DIMENSION CLEANER
# ==========================================
def parse_dim(text):
    if not text: return 0.0
    clean = str(text).replace('"', '').replace("'", "").strip()
    try:
        # Try Feet-Inch: 3-0 or 3 0
        parts = re.split(r'[-\s]+', clean)
        ft = float(parts[0])
        inch = float(parts[1]) if len(parts) > 1 else 0
        return ft + inch/12.0
    except: return 0.0

# ==========================================
# 4. EXECUTE
# ==========================================
def execute(pdf_path, pages):
    print("--- [Phase 2] Smart Extraction ---")
    doc = fitz.open(pdf_path)
    
    library = {"windows": {}, "doors": {}}
    debug_crops = [] # For main.py visualization

    for p_meta in pages:
        p_num = p_meta['page']
        p_type = p_meta['type']
        
        # Use high-res for reading text
        page = doc[p_num-1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # Logic Switch based on Phase 1 Classification
        items = []
        if p_type == "Schedule":
            print(f"  P{p_num}: Reading Table (Schedule Mode)...")
            items = extract_schedule_data(img)
            debug_crops.append((img, f"P{p_num} Schedule Table"))
            
        elif p_type == "Type_Definition":
            print(f"  P{p_num}: Reading Visuals (Legend Mode)...")
            items = extract_visual_data(img)
            debug_crops.append((img, f"P{p_num} Visual Legend"))
            
        elif p_type == "Floor_Plan":
            # Optional: Add Keynote Extraction logic here if needed later
            continue

        # Merge & Clean Logic
        for item in items:
            raw_mark = item.get("mark", "")
            if not raw_mark: continue
            
            # Normalize Tag
            mark = raw_mark.strip().upper()
            
            # --- THE GARBAGE FILTER ---
            # 1. Reject specific keywords often found in detail columns
            if mark in ["TYP", "SIM", "OPP", "DTL", "NTS", "REF", "HM", "ALUM"]: continue
            if "DETAIL" in mark or "NOTE" in mark: continue
            
            # 2. Reject tags that are too long (likely sentences)
            if len(mark) > 8: continue 
            
            # 3. Reject tags starting with 'TYP-' unless it's a known convention
            if mark.startswith("TYP-") and len(mark) > 6: continue

            # Parse Dims
            w = parse_dim(item.get("width_str"))
            h = parse_dim(item.get("height_str"))
            cat = item.get("category", "window").lower()
            
            # Only add if valid dimensions found
            if w > 0 and h > 0:
                entry = {
                    "width": w, 
                    "height": h, 
                    "raw_w": item.get("width_str"), 
                    "raw_h": item.get("height_str"),
                    "category": cat
                }
                
                # Assign to library buckets
                # Note: "opening" or "louver" goes to windows bucket for now
                if "door" in cat:
                    library["doors"][mark] = entry
                else:
                    library["windows"][mark] = entry
                
    print(f"  Cataloged {len(library['windows'])} Windows, {len(library['doors'])} Doors.")
    return library, debug_crops