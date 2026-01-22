import fitz  # PyMuPDF
import ollama
import io
import json
import re
import math
import logging
from PIL import Image, ImageDraw

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "qwen3-vl:30b-a3b-instruct"

# ==========================================
# HELPER: BULLETPROOF JSON PARSER
# ==========================================
def extract_json(content):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r"(\{.*\})", content, re.DOTALL)
        if match: return json.loads(match.group(1))
        clean = content.strip().replace("json", "").replace("```", "")
        return json.loads(clean)
    except:
        return None

# ==========================================
# AGENT 1: SEGREGATE (VIEW FINDER)
# ==========================================
def agent_segregate_views(page_image):
    logger.info("--- AGENT 1: Segregating Views ---")
    byte_arr = io.BytesIO()
    page_image.save(byte_arr, format='PNG')
    img_bytes = byte_arr.getvalue()

    prompt = """
    Analyze this sheet.
    TASK: Return bounding boxes for distinct drawings (Elevation, Plan, etc).
    OUTPUT JSON: {"views": [{"label": "Name", "box_1000": [x0, y0, x1, y1]}]}
    """
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt, 'images': [img_bytes]}])
        data = extract_json(response['message']['content'])
        if data: return data.get("views", [])
    except Exception as e:
        logger.error(f"Segregation Error: {e}")
    
    return [{"label": "Full Page Fallback", "box_1000": [0,0,1000,1000]}]

# ==========================================
# AGENT 2: STRATEGIST (ORIENTATION)
# ==========================================
def agent_decide_orientation(view_image):
    logger.info("--- AGENT 2: Deciding Orientation ---")
    byte_arr = io.BytesIO()
    view_image.save(byte_arr, format='PNG')
    img_bytes = byte_arr.getvalue()

    prompt = """
    Act as an Architect.
    TASK: Determine the best orientation to calibrate the scale of this drawing.
    
    OPTIONS:
    1. "HORIZONTAL": If you see clear horizontal dimension strings.
    2. "VERTICAL": If you see clear vertical dimension strings (e.g. Floor Heights).
    
    OUTPUT JSON: {"orientation": "VERTICAL" or "HORIZONTAL", "reason": "..."}
    """
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt, 'images': [img_bytes]}])
        data = extract_json(response['message']['content'])
        if data: return data.get("orientation", "VERTICAL").upper()
    except Exception as e:
        logger.error(f"Strategy Error: {e}")
    return "VERTICAL"

# ==========================================
# PYTHON: GLOBAL VECTOR EXTRACTION
# ==========================================
def extract_vectors_global(page, view_rect, orientation, view_logs):
    logger.info(f"--- PYTHON: Scanning view for {orientation} vectors ---")
    
    words = page.get_text("words")
    anchors = []
    strict_pattern = re.compile(r".*\d.*['\"-].*")
    forbidden = ["EL", "T.O.", "SIM", "TYP", "LEVEL", "CL", "GRID"]

    # 1. Collect Anchors
    for w in words:
        pt = fitz.Point(w[0], w[1])
        if view_rect.contains(pt):
            clean_text = w[4].strip()
            if strict_pattern.match(clean_text):
                if not any(bad in clean_text.upper() for bad in forbidden):
                    anchors.append({
                        "text": clean_text, 
                        "center": fitz.Point((w[0]+w[2])/2, (w[1]+w[3])/2)
                    })
    
    if not anchors:
        view_logs.append("  Debug: No valid text anchors found.")
        return []

    # 2. Extract Lines
    drawings = page.get_drawings()
    candidates = []
    
    for path in drawings:
        for item in path['items']:
            if item[0] == 'l':
                p1, p2 = item[1], item[2]
                if not (view_rect.contains(p1) and view_rect.contains(p2)): continue
                
                length = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
                if length < 25: continue 
                
                is_horz = abs(p1.y - p2.y) < 2.0
                is_vert = abs(p1.x - p2.x) < 2.0
                
                if orientation == "VERTICAL" and not is_vert: continue
                if orientation == "HORIZONTAL" and not is_horz: continue
                if not (is_horz or is_vert): continue

                mid = fitz.Point((p1.x+p2.x)/2, (p1.y+p2.y)/2)
                best_dist = 10000
                best_anchor = None
                
                for anchor in anchors:
                    d = math.sqrt((mid.x-anchor['center'].x)**2 + (mid.y-anchor['center'].y)**2)
                    if d < 120: 
                        if d < best_dist:
                            best_dist = d
                            best_anchor = anchor
                        
                if best_anchor:
                    candidates.append({
                        "p1": p1, "p2": p2, "len": length,
                        "anchor_text": best_anchor['text'],
                        "score": best_dist
                    })
    
    view_logs.append(f"  Debug: Found {len(candidates)} {orientation} candidates.")
    candidates.sort(key=lambda x: x['score'])
    
    unique = []
    seen = set()
    for c in candidates:
        k = f"{round(c['len'],1)}_{c['anchor_text']}"
        if k not in seen:
            unique.append(c)
            seen.add(k)
            
    return unique[:6]

# ==========================================
# AGENT 3: CHAIN-OF-THOUGHT VALIDATOR
# ==========================================
def agent_validate_vector(img_bytes, vector, orientation):
    prompt = f"""
    You are a Calibration Auditor.
    CONTEXT:
    - We are validating a {orientation} Dimension.
    - Anchor Text is "{vector['anchor_text']}".
    
    LOOK AT THE BLUE HIGHLIGHTED LINE.
    
    TASK: Determine if this is a valid dimension line.
    
    CRITERIA:
    1. Does the line connect two specific points (e.g. floor lines, grid lines)?
    2. Is the text clearly associated with this line?
    
    OUTPUT JSON: 
    {{ 
      "verdict": "VALID" or "INVALID", 
      "reason": "..." 
    }}
    """
    logger.info("--- AGENT 3: Validating Vector ---")
    
    for _ in range(2): 
        try:
            response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt, 'images': [img_bytes]}])
            content = response['message']['content']
            data = extract_json(content)
            if data: return data
        except: pass
            
    return {"verdict": "ERROR", "reason": "VLM Failed"}

# ==========================================
# AGENT 4: READER & PARSER
# ==========================================
def agent_read_scale(img_bytes):
    prompt = "Read the dimension text associated with the BLUE LINE. Return ONLY the value (e.g. 4'-0\")."
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt, 'images': [img_bytes]}])
        return response['message']['content'].strip()
    except: return ""

def parse_feet(dim_str):
    if not dim_str: return None
    clean = dim_str.replace("’", "'").replace("”", '"').strip()
    match = re.search(r"(\d+)'\s*-?\s*(\d+)?(?:[ ]+(\d+)/(\d+))?", clean)
    if match:
        f = float(match.group(1))
        i = float(match.group(2)) if match.group(2) else 0
        num = float(match.group(3)) if match.group(3) else 0
        den = float(match.group(4)) if match.group(4) else 1
        return f + (i + (num/den))/12.0
    
    match_dec = re.search(r"(\d+)'\s*(\d+(?:\.\d+)?)\"", clean)
    if match_dec:
        f = float(match_dec.group(1))
        i = float(match_dec.group(2))
        return f + i/12.0
        
    return None

def generate_highlighted_image(page, vector):
    margin = 150 
    p1, p2 = vector['p1'], vector['p2']
    x0 = max(0, min(p1.x, p2.x) - margin)
    y0 = max(0, min(p1.y, p2.y) - margin)
    x1 = min(page.rect.width, max(p1.x, p2.x) + margin)
    y1 = min(page.rect.height, max(p1.y, p2.y) + margin)
    crop = fitz.Rect(x0, y0, x1, y1)

    pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), clip=crop)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    draw = ImageDraw.Draw(img)
    sx, sy = img.width/crop.width, img.height/crop.height
    
    draw.line([
        (p1.x-crop.x0)*sx, (p1.y-crop.y0)*sy,
        (p2.x-crop.x0)*sx, (p2.y-crop.y0)*sy
    ], fill="blue", width=6)
    
    return img

# ==========================================
# 5. PIPELINE ENTRY POINT
# ==========================================
def execute(pdf_path, elevation_pages):
    print(f"--- [Phase 4] Chain-of-Thought Calibration ---")
    doc = fitz.open(pdf_path)
    scale_data = {} 
    debug_artifacts = [] 
    
    for page_num in elevation_pages:
        print(f"  > Calibrating Page {page_num}...")
        try:
            page = doc[page_num - 1]
            scale_data[page_num] = {}
            pdf_w, pdf_h = page.rect.width, page.rect.height
            
            pix = page.get_pixmap(matrix=fitz.Matrix(1,1))
            full_img = Image.open(io.BytesIO(pix.tobytes("png")))
            views_data = agent_segregate_views(full_img)
            
            for i, v_data in enumerate(views_data):
                label = v_data.get("label", f"View {i+1}")
                box = v_data.get("box_1000", [0,0,1000,1000])
                view_rect = fitz.Rect((box[0]/1000)*pdf_w, (box[1]/1000)*pdf_h, (box[2]/1000)*pdf_w, (box[3]/1000)*pdf_h)
                
                pix_v = page.get_pixmap(matrix=fitz.Matrix(1,1), clip=view_rect)
                view_img = Image.open(io.BytesIO(pix_v.tobytes("png")))
                
                view_logs = []
                orientation = agent_decide_orientation(view_img)
                vectors = extract_vectors_global(page, view_rect, orientation, view_logs)
                
                success_flag = False
                
                for j, vec in enumerate(vectors):
                    img = generate_highlighted_image(page, vec)
                    byte_arr = io.BytesIO()
                    img.save(byte_arr, format='PNG')
                    img_bytes = byte_arr.getvalue()
                    
                    val_res = agent_validate_vector(img_bytes, vec, orientation)
                    verdict = val_res.get("verdict", "INVALID")
                    
                    status = f"{label} Candidate {j+1}: {verdict}"
                    debug_artifacts.append((img, f"P{page_num} {status}"))
                    
                    if verdict == "VALID":
                        val_str = agent_read_scale(img_bytes)
                        feet = parse_feet(val_str)
                        if feet and feet > 0.5:
                            scale = vec['len'] / feet
                            scale_data[page_num][label] = scale
                            print(f"    - {label}: SUCCESS. Scale = {scale:.2f} pts/ft (Ref: {val_str})")
                            debug_artifacts[-1] = (img, f"P{page_num} {label} Scale={scale:.2f} (Ref: {val_str})")
                            success_flag = True
                            break 
                
                if not success_flag:
                    print(f"    - {label}: FAILED. All candidates rejected.")

        except Exception as e:
            print(f"    [Error] P{page_num}: {e}")

    return scale_data, debug_artifacts