"""
Fascia_Gemini.py — Full single-file pipeline for Fascia components.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import io, os, re, json, sys, time, shutil, logging, traceback, uuid
import fitz
from PIL import Image, ImageDraw
from google import genai
from google.genai import types, errors as genai_errors
from google.api_core.exceptions import TooManyRequests, ServiceUnavailable
from pipeline.runner import update_heartbeat, is_cancelled

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(__file__), "fascia_pipeline.log")
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

def log_error(ctx, exc):
    logging.error(f"{ctx}\n{traceback.format_exc()}")
    print(f"[ERROR] {ctx}: {exc}")

def log_response(ctx, text):
    logging.info(f"[RESPONSE] {ctx}:\n{text}\n" + "-"*40)

# ── Gemini client ─────────────────────────────────────────────────────────────
PROJECT_ID  = "neurainternalnoorg"
LOCATION    = "global"
MODEL_NAME  = "gemini-3-pro-preview"
_ZOOM_MODEL = "gemini-3-flash-preview"
_FASCIA_MODEL = "gemini-3-pro-preview"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# ── Shared helpers ────────────────────────────────────────────────────────────

def _is_rate_limit(e):
    s = str(e)
    return "429" in s or "RESOURCE_EXHAUSTED" in s or "Resource exhausted" in s

_MAX_RETRY_WAIT = 60   # never sleep longer than 60s per retry
_HEARTBEAT_INTERVAL = 30  # send heartbeat every 30s during waits

def _heartbeat_sleep(seconds, run_id=None):
    """Sleep in chunks, sending heartbeat every _HEARTBEAT_INTERVAL seconds."""
    elapsed = 0
    while elapsed < seconds:
        if is_cancelled():
            return  # stop sleeping if pipeline cancelled
        chunk = min(_HEARTBEAT_INTERVAL, seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        update_heartbeat(run_id)

def _gemini_retry(model, contents, config, retries=3, wait=10, label="", run_id=None):
    for attempt in range(1, retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            return resp
        except (TooManyRequests, ServiceUnavailable) as e:
            if attempt == retries: raise
            w = min(wait, _MAX_RETRY_WAIT)
            print(f"[{label}] Rate limit attempt {attempt}/{retries} — retry in {w}s")
            _heartbeat_sleep(w, run_id); wait *= 2
        except genai_errors.ClientError as e:
            if _is_rate_limit(e):
                if attempt == retries: raise
                w = min(wait, _MAX_RETRY_WAIT)
                print(f"[{label}] Rate limit attempt {attempt}/{retries} — retry in {w}s")
                _heartbeat_sleep(w, run_id); wait *= 2
            else:
                log_error(label, e); raise
        except Exception as e:
            if _is_rate_limit(e):
                if attempt == retries: raise
                w = min(wait, _MAX_RETRY_WAIT)
                print(f"[{label}] Rate limit attempt {attempt}/{retries} — retry in {w}s")
                _heartbeat_sleep(w, run_id); wait *= 2
            else:
                log_error(label, e); raise

def _get_text(response):
    if not response: return None
    if response.text: return response.text
    try:
        parts = [p.text for c in (response.candidates or [])
                 for p in (c.content.parts or []) if getattr(p, "text", None)]
        return " ".join(parts) if parts else None
    except Exception: return None


def _pdf_page_to_image_part(full_page_path: str, page_number: int = 0):
    """
    Rasterize a PDF page to PNG bytes and return a types.Part.
    Used as a fallback when no crop image is available, because Gemini
    rejects mime_type=application/pdf with response_mime_type=application/json.
    Returns None if rasterisation fails.
    """
    try:
        doc = fitz.open(full_page_path)
        page = doc[min(page_number, len(doc) - 1)]
        mat = fitz.Matrix(2, 2)  # 2x zoom for readable resolution
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        doc.close()
        return types.Part.from_bytes(data=png_bytes, mime_type="image/png")
    except Exception as e:
        print(f"[_pdf_page_to_image_part] Failed to rasterize {full_page_path}: {e}")
        return None


def _repair_json(text):
    text = text.rstrip()
    if text.startswith("{") and not text.endswith("}"):
        if text.count('"') % 2 == 1: text += '"'
        text = text.rstrip(",").rstrip() + "}"
    return text

def _parse_json(raw, required_keys=None, phase=""):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$",           "", raw, flags=re.IGNORECASE)
    raw = raw.strip()
    for candidate in (raw, _repair_json(raw)):
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if required_keys:
                    for k in required_keys:
                        parsed.setdefault(k, None)
                return parsed
            except json.JSONDecodeError: continue
    raise RuntimeError(f"[{phase} ERROR] Invalid JSON.\n\nRaw:\n{raw}")

class _ModelWrapper:
    def generate_content(self, contents):
        return _gemini_retry(MODEL_NAME, contents,
                             types.GenerateContentConfig(temperature=0.2, top_p=0.95,
                                                         max_output_tokens=4096),
                             retries=7, label="generate_with_retry")

model = _ModelWrapper()

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH TEXT
# ══════════════════════════════════════════════════════════════════════════════

def search_text_in_pdf(file_path, search_text, page_number=0, instance_index=0):
    doc = fitz.open(file_path)
    page = doc[page_number]
    all_drawings = page.get_drawings()
    text_instances = page.search_for(search_text)
    
    result = {
        "found": False,
        "text_rect": None,
        "drawing_rect": None,
        "crop_path": None,
        "full_page_path": None
    }

    if not text_instances or instance_index >= len(text_instances):
        doc.close(); return result

    result["found"] = True
    inst = text_instances[instance_index]
    result["text_rect"] = [inst.x0, inst.y0, inst.x1, inst.y1]
    
    # Highlight text
    shape = page.new_shape()
    shape.draw_rect(inst)
    shape.finish(color=(1, 0, 0), width=2)
    shape.commit()

    # Find nearby drawing
    target_drawing_rect = None
    for drawing in all_drawings:
        if (inst.x0 < drawing["rect"][2] and inst.x1 > drawing["rect"][0] and 
            inst.y0 < drawing["rect"][3] and inst.y1 > drawing["rect"][1]):
            # Refined: We no longer draw heuristic boxes here as they are often incorrect.
            # We just track the rect to inform the VLM zoom window.
            target_drawing_rect = drawing["rect"]
            break
            
    if target_drawing_rect:
        result["drawing_rect"] = [target_drawing_rect[0], target_drawing_rect[1], target_drawing_rect[2], target_drawing_rect[3]]
    
    # Save full highlighted page
    img = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    unique_id = uuid.uuid4().hex[:8]
    full_pdf_path = f"fascia_extracted_{unique_id}_{page_number + 1}.pdf"
    
    pdf_output = fitz.open()
    pdf_page = pdf_output.new_page(width=img.width, height=img.height)
    pdf_page.insert_image(pdf_page.rect, pixmap=img)
    pdf_output.save(full_pdf_path)
    pdf_output.close()
    result["full_page_path"] = full_pdf_path
    
    # Fallback Crop Generation
    try:
        target_r = result["drawing_rect"] if result["drawing_rect"] else result["text_rect"]
        if target_r:
            pad = 200
            crop_rect = fitz.Rect(target_r[0]-pad, target_r[1]-pad, target_r[2]+pad, target_r[3]+pad)
            crop_rect = crop_rect & page.rect
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=crop_rect)
            crop_path = f"fascia_search_crop_{uuid.uuid4().hex[:6]}_{page_number+1}.png"
            pix.save(crop_path)
            result["crop_path"] = crop_path
    except Exception: pass

    doc.close()
    return result

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 TOOL — VLM Zoom
# ══════════════════════════════════════════════════════════════════════════════

def crop_image(ymin: int, xmin: int, ymax: int, xmax: int, keynote_symbol: str = ""):
    return "Cropping image..."

def _render_region_to_jpeg(pdf_path: str, region: fitz.Rect):
    doc = fitz.open(pdf_path)
    page = doc[0]
    region = region & page.rect
    scale = min(3000 / max(region.width, region.height), 3.0)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=region)
    data = pix.tobytes("jpeg")
    w, h = pix.width, pix.height
    doc.close()
    return data, w, h, scale

def get_vlm_crop(pdf_path: str, search_text: str, text_rect=None, drawing_rect=None):
    doc = fitz.open(pdf_path)
    page = doc[0]
    window = page.rect
    win_x0, win_y0 = window.x0, window.y0
    jpeg_bytes, img_w, img_h, sub_scale = _render_region_to_jpeg(pdf_path, window)
    doc.close()

    text_hint = ""
    if text_rect:
        tx0, ty0, tx1, ty1 = (int((text_rect[0]-win_x0)*sub_scale), int((text_rect[1]-win_y0)*sub_scale),
                             int((text_rect[2]-win_x0)*sub_scale), int((text_rect[3]-win_y0)*sub_scale))
        text_hint = f" The label '{search_text}' was found at approx x=[{tx0}–{tx1}], y=[{ty0}–{ty1}]."

    prompt = (
        f"This image ({img_w}×{img_h} px) is a full architectural construction sheet.{text_hint}\n\n"
        f"TASK: Find the boundary of the ENTIRE architectural detail drawing block associated with '{search_text}'.\n\n"
        f"STRATEGY:\n"
        f"1. Locate '{search_text}' in the sheet using the hint above.\n"
        f"2. Decide: is it inside a 'Notes/Keynotes/Legend' list — or directly inside a diagram?\n"
        f"3. IF in Notes/Keynotes/Legend:\n"
        f"   a. READ EACH ROW: [number] [description text].\n"
        f"   b. Find the ONE row whose description contains '{search_text}'.\n"
        f"   c. Record that row's number/symbol — that is your keynote.\n"
        f"4. Scan the ENTIRE SHEET for the diagram containing that keynote bubble.\n"
        f"5. IF '{search_text}' is directly in a diagram, focus on that diagram.\n\n"
        f"HOW TO CROP — STRICT BOUNDARY RULES:\n"
        f"- You MUST capture the COMPLETE diagram. NEVER clip any part of it.\n"
        f"- ymin: Set 30-50 px ABOVE the topmost element (dimension lines, extension lines, keynote bubbles, or drawing frame).\n"
        f"- ymax: Set 30-50 px BELOW the Drawing Title/SCALE text at the bottom of the detail.\n"
        f"- xmin: Set 30-50 px LEFT of the leftmost element (dimension line, callout, or drawing frame edge).\n"
        f"- xmax: Set 30-50 px RIGHT of the rightmost element (dimension line, callout, or drawing frame edge).\n"
        f"- Include ALL: dimension lines, extension lines, keynote bubbles, leader lines, section cut marks, \n"
        f"  hatching/patterns, material annotations, elevation markers, and the detail title.\n"
        f"- When in doubt, crop LARGER rather than smaller — it is far worse to clip part of the \n"
        f"  diagram than to include a small amount of surrounding whitespace.\n\n"
        f"CRITICAL RULES:\n"
        f"- ONE bounding box only.\n"
        f"- If from Notes list: crop the DIAGRAM the keynote points to, NOT the Notes section.\n"
        f"- Crop MUST include the diagram's title/SCALE at the bottom.\n"
        f"- Crop MUST include ALL dimension lines and callouts that belong to this detail.\n"
        f"- Pass the keynote number/symbol as 'keynote_symbol' (empty if not from notes).\n"
        f"- Coordinates: integers, x∈[0,{img_w}], y∈[0,{img_h}].\n"
        f"- NEVER return a crop that is less than 10% of the image area — if the diagram is \n"
        f"  that small, you are likely missing parts of it."
    )

    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=_ZOOM_MODEL,
                contents=[prompt, types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    tools=[crop_image],
                    tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                        mode="ANY", allowed_function_names=["crop_image"])),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            if response.function_calls:
                fc = response.function_calls[0]
                args = fc.args
                sy0, sx0, sy1, sx1 = int(args["ymin"]), int(args["xmin"]), int(args["ymax"]), int(args["xmax"])
                keynote_symbol = args.get("keynote_symbol", "")
                coords = {
                    "ymin": int(win_y0 + sy0/sub_scale), "xmin": int(win_x0 + sx0/sub_scale),
                    "ymax": int(win_y0 + sy1/sub_scale), "xmax": int(win_x0 + sx1/sub_scale),
                }
                return coords, keynote_symbol
            return None, ""
        except Exception as e:
            if _is_rate_limit(e) and attempt < 3:
                w = min(10 * (2**(attempt-1)), _MAX_RETRY_WAIT)
                print(f"[VLM Zoom] Rate limit attempt {attempt}/3 — retry in {w}s")
                _heartbeat_sleep(w)
            else: return None, ""

_CROP_PAD = 15   # extra PDF-pt padding around VLM coords
_MIN_CROP = 120  # minimum crop dimension in PDF points


def execute_crop(pdf_path: str, coords: dict, output_path: str) -> bool:
    """Crop PDF page at coords with safety padding + minimum size enforcement."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    pw, ph = page.rect.width, page.rect.height

    # Apply safety padding
    x0 = max(0,  coords["xmin"] - _CROP_PAD)
    y0 = max(0,  coords["ymin"] - _CROP_PAD)
    x1 = min(pw, coords["xmax"] + _CROP_PAD)
    y1 = min(ph, coords["ymax"] + _CROP_PAD)

    # Enforce minimum crop size
    cw, ch = x1 - x0, y1 - y0
    if cw < _MIN_CROP:
        cx = (x0 + x1) / 2
        x0 = max(0,  cx - _MIN_CROP / 2)
        x1 = min(pw, cx + _MIN_CROP / 2)
    if ch < _MIN_CROP:
        cy = (y0 + y1) / 2
        y0 = max(0,  cy - _MIN_CROP / 2)
        y1 = min(ph, cy + _MIN_CROP / 2)

    if x1 <= x0 or y1 <= y0: doc.close(); return False
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=fitz.Rect(x0, y0, x1, y1))
    pix.save(output_path); doc.close()
    return True

def get_keynote_dimension(pdf_path: str, keynote_symbol: str, search_text: str) -> str:
    if not keynote_symbol: return ""
    try:
        jpeg_bytes, img_w, img_h, _ = _render_region_to_jpeg(pdf_path, fitz.open(pdf_path)[0].rect)
        prompt = (f"Full sheet ({img_w}×{img_h} px). Find keynote '{keynote_symbol}' ('{search_text}'). "
                  f"Return ONLY its dimension (e.g. '3/4\"') or NONE.")
        resp = client.models.generate_content(
            model=_ZOOM_MODEL, contents=[prompt, types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")],
            config=types.GenerateContentConfig(temperature=0.0))
        raw = (resp.text or "").strip()
        return "" if raw.upper() == "NONE" or not raw else raw
    except Exception: return ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1b — Component Highlighting
# ══════════════════════════════════════════════════════════════════════════════

_ZOOM_PAD = 120

def _expand_box(b, img_w, img_h):
    """Expand thin boxes (height or width ≤40px) to be visually clear."""
    bw, bh = b["xmax"]-b["xmin"], b["ymax"]-b["ymin"]
    cy, cx  = (b["ymin"]+b["ymax"])//2, (b["xmin"]+b["xmax"])//2
    SPAN    = 250
    if bh <= 40:
        return [max(0,cx-SPAN), max(0,cy-max(bh//2,4)), min(img_w,cx+SPAN), min(img_h,cy+max(bh//2,4))]
    if bw <= 40:
        return [max(0,cx-max(bw//2,4)), max(0,cy-SPAN), min(img_w,cx+max(bw//2,4)), min(img_h,cy+SPAN)]
    return [b["xmin"], b["ymin"], b["xmax"], b["ymax"]]

def _fc_to_bbox(fc_args):
    """Convert function call args to a bbox dict."""
    return {k: int(fc_args[k]) for k in ("ymin","xmin","ymax","xmax")}

def locate_arrow_tip(ymin: int, xmin: int, ymax: int, xmax: int):
    """
    Call this tool with a small tight bbox around the exact pixel location
    where the FASCIA leader line ENDS (the arrowhead tip in the drawing).
    Coordinates are pixels relative to the image shown to you.

    Args:
        ymin: Top pixel of the arrowhead region.
        xmin: Left pixel of the arrowhead region.
        ymax: Bottom pixel of the arrowhead region.
        xmax: Right pixel of the arrowhead region.
    """
    return "Arrow tip location received."

def highlight_fascia(ymin: int, xmin: int, ymax: int, xmax: int):
    """
    Call this tool once for EACH separate location where the fascia material
    is present in the image shown to you.
    Coordinates are pixels relative to the FULL cropped image shown to you.

    Args:
        ymin: Top pixel of one fascia occurrence.
        xmin: Left pixel of one fascia occurrence.
        ymax: Bottom pixel of one fascia occurrence.
        xmax: Right pixel of one fascia occurrence.
    """
    return "Fascia location recorded."

def locate_all_occurrences(ymin: int, xmin: int, ymax: int, xmax: int):
    return "Occurrence location recorded."

def _call_vlm(prompt, image_bytes, mime_type="image/png", tools=None, tool_config=None, afc_off=False, label="Phase 1b", run_id=None):
    update_heartbeat(run_id)
    cfg = dict(temperature=0.0)
    if tools: cfg["tools"] = tools
    if tool_config: cfg["tool_config"] = tool_config
    if afc_off: cfg["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    for attempt in range(1, 4):
        try:
            return client.models.generate_content(model=_FASCIA_MODEL, contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)], config=types.GenerateContentConfig(**cfg))
        except Exception as e:
            if _is_rate_limit(e) and attempt < 3:
                w = min(10 * (2**(attempt-1)), _MAX_RETRY_WAIT)
                print(f"[{label}] Rate limit — retry in {w}s")
                _heartbeat_sleep(w, run_id)
            else:
                print(f"[{label}] Error: {e}"); return None
    return None

def _zoom_region(image_bytes: bytes, bbox: dict, pad: int = _ZOOM_PAD) -> tuple:
    """
    Crops bbox+padding from image_bytes.
    Returns (jpeg_bytes, off_x, off_y, zoom_w, zoom_h).
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        W, H = img.size
        x0 = max(0, bbox["xmin"] - pad)
        y0 = max(0, bbox["ymin"] - pad)
        x1 = min(W, bbox["xmax"] + pad)
        y1 = min(H, bbox["ymax"] + pad)
        crop = img.crop((x0, y0, x1, y1))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), x0, y0, crop.width, crop.height

def _crop_and_annotate_occurrence(image_bytes, bbox, base_path, idx, color="red", pad=80):
    with Image.open(io.BytesIO(image_bytes)) as img:
        W, H = img.size
        x0, y0 = max(0, bbox["xmin"]-pad), max(0, bbox["ymin"]-pad)
        x1, y1 = min(W, bbox["xmax"]+pad), min(H, bbox["ymax"]+pad)
        crop = img.crop((x0, y0, x1, y1)).copy()
    draw = ImageDraw.Draw(crop)
    draw.rectangle([bbox["xmin"]-x0, bbox["ymin"]-y0, bbox["xmax"]-x0, bbox["ymax"]-y0], outline=color, width=5)
    out = f"{base_path}_occurrence_{idx}.png"
    crop.save(out, format="PNG")
    return out

def _draw_boxes_on_image(annotated_path, bboxes, img_w, img_h):
    with Image.open(annotated_path) as im:
        draw = ImageDraw.Draw(im)
        for b in bboxes:
            draw.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline="red", width=5)
        im.save(annotated_path)

def run_phase1b(image_path: str, keyword: str, keynote_symbol: str = "", run_id: str = None):
    """
    PHASE 1b: Fascia Highlighting.
    Step 1  Arrow Info  : VLM traces FASCIA arrows and describes material at tip (text).
    Step 2  Locate Tip  : locate_arrow_tip tool call -> tight bbox -> zoom target.
    Step 3  Zoom+Verify : zoom crop -> VLM verifies exact material description (text).
    Step 4  Full BBox   : full image scan -> highlight_fascia for every occurrence.
    Step 5  Annotate    : draw plain red rectangles on ALL occurrences -> return.
    """
    update_heartbeat(run_id)
    if not image_path or not os.path.exists(image_path):
        print(f"[Phase 1b] Input image not found: {image_path}")
        return None, [], "", []

    print(f"--- [Phase 1b] Identifying Fascia in {os.path.basename(image_path)} ---")

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    with Image.open(io.BytesIO(image_bytes)) as img:
        width, height = img.size

    base, ext = os.path.splitext(image_path)
    annotated_path = f"{base}_annotated{ext or '.png'}"
    try:
        shutil.copy2(image_path, annotated_path)
    except Exception as e:
        print(f"[Phase 1b] Error initialising annotated image: {e}")
        return None, [], "", []

    tip_cfg = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
        mode="ANY", allowed_function_names=["locate_arrow_tip"]))
    fascia_cfg = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
        mode="ANY", allowed_function_names=["highlight_fascia"]))

    # -- Step 1: Arrow Info ---------------------------------------------------
    # Ask VLM where the FASCIA arrows point - text description on the full
    # cropped image. This gives us the general region to zoom into.
    print("[Phase 1b] Step 1: Identifying where FASCIA arrows end...")
    target_label = f"keynote bubble '{keynote_symbol}'" if keynote_symbol else f"the word '{keyword}'"
    step1_prompt = (
        f"This is a construction detail drawing ({width}x{height} px).\n"
        f"Find {target_label} and trace every leader line / arrow from it.\n\n"
        "Answer ONLY:\n"
        "  1. How many leader lines come from it?\n"
        "  2. Where does each arrow TIP end? Give approximate pixel coords "
        "(e.g. 'around x=200-350, y=80-120') and describe the component there.\n"
        "  3. Describe the material/texture at the arrow endpoint in detail "
        "(grain, pattern, thickness, orientation).\n"
        "Do NOT describe anything else."
    )
    s1 = _call_vlm(step1_prompt, image_bytes, mime_type="image/png",
                   label="Phase 1b Step 1", run_id=run_id)
    if not s1:
        return None, [], "", []

    arrow_info = getattr(s1, "text", "") or ""
    log_response("Phase 1b Arrow Info", arrow_info)
    print(f"[Phase 1b] Step 1 done. Info: {arrow_info[:150]}...")

    # -- Step 2: Locate arrow tip (tool call -> zoom target) ------------------
    print("[Phase 1b] Step 2: Pinpointing arrow tip location...")
    step2_prompt = (
        f"This is a construction detail drawing ({width}x{height} px).\n\n"
        f"Arrow analysis found:\n---\n{arrow_info}\n---\n\n"
        f"TASK: Trace the leader line(s) to the very TIP of the arrow.\n"
        f"Call `locate_arrow_tip` with a tight box around just the arrowhead endpoint.\n"
        f"If multiple arrows, give a box that covers all their endpoints together.\n"
        f"Coordinates: integers, x in [0,{width}], y in [0,{height}]."
    )
    s2 = _call_vlm(step2_prompt, image_bytes, mime_type="image/png",
                   tools=[locate_arrow_tip], tool_config=tip_cfg,
                   afc_off=True, label="Phase 1b Step 2", run_id=run_id)

    if not s2 or not s2.function_calls:
        print("[Phase 1b] Step 2: No locate_arrow_tip call - cannot zoom.")
        return None, [], arrow_info, []

    tip_args = s2.function_calls[0].args
    tip_bbox = {k: int(tip_args[k]) for k in ("ymin", "xmin", "ymax", "xmax")}
    log_response("Phase 1b Tip BBox", str(tip_bbox))
    print(f"[Phase 1b] Step 2: Arrow tip bbox = {tip_bbox}")

    # -- Step 3: Zoom & Verify ------------------------------------------------
    # Crop the arrow-tip region and send to VLM for detailed material verification.
    # This verified description will guide the full-image scan.
    print("[Phase 1b] Step 3: Zooming in to verify location and material...")
    try:
        zoom_bytes, off_x, off_y, zoom_w, zoom_h = _zoom_region(
            image_bytes, tip_bbox, pad=_ZOOM_PAD
        )
    except Exception as e:
        print(f"[Phase 1b] Step 3: Zoom failed: {e}")
        return None, [], arrow_info, []

    step3_prompt = (
        f"This is a zoomed-in crop ({zoom_w}x{zoom_h} px) from a construction drawing, "
        f"centred on where a FASCIA arrow ends.\n\n"
        f"Earlier context:\n---\n{arrow_info}\n---\n\n"
        f"TASK: You must anchor your answer to the EXACT pixel location where the arrow tip lands.\n"
        f"  1. Confirm the EXACT pixel location of the arrow endpoint in this image - "
        f"do not approximate or guess. Point to the precise spot.\n"
        f"  2. Describe ONLY the material/element at THAT EXACT POINT - its texture, "
        f"grain pattern, orientation, thickness, colour, and any visible edges.\n"
        f"  3. Give a description precise enough that ONLY this material would match - "
        f"not any nearby or visually similar material.\n"
        f"STRICT RULE: Do not describe anything that the arrow is NOT directly touching."
    )
    s3 = _call_vlm(step3_prompt, zoom_bytes, mime_type="image/jpeg",
                   label="Phase 1b Step 3", run_id=run_id)

    if not s3:
        print("[Phase 1b] Step 3: Verification failed.")
        return None, [], arrow_info, []

    verified_material = getattr(s3, "text", "") or ""
    log_response("Phase 1b Verified Material", verified_material)
    print(f"[Phase 1b] Step 3: Material verified: {verified_material[:150]}...")

    # -- Step 4: Full-Image BBox ----------------------------------------------
    # Send FULL cropped image + verified material description.
    # VLM calls highlight_fascia once for EVERY location that material appears.
    print("[Phase 1b] Step 4: Scanning full image for ALL fascia occurrences...")
    step4_prompt = (
        f"This is a full construction detail drawing ({width}x{height} px).\n\n"
        f"CONFIRMED ARROW LOCATION: The FASCIA arrow tip was at\n"
        f"  y=[{tip_bbox['ymin']}-{tip_bbox['ymax']}], x=[{tip_bbox['xmin']}-{tip_bbox['xmax']}].\n\n"
        f"CONFIRMED MATERIAL at that arrow tip:\n---\n{verified_material}\n---\n\n"
        f"TASK - follow these rules STRICTLY:\n"
        f"  STEP A: First, locate the confirmed arrow tip region in the full image "
        f"(y=[{tip_bbox['ymin']}-{tip_bbox['ymax']}], x=[{tip_bbox['xmin']}-{tip_bbox['xmax']}]).\n"
        f"  STEP B: Draw the FIRST bbox around the fascia piece that the arrow is directly touching.\n"
        f"  STEP C: Then scan the ENTIRE image for other pieces of the IDENTICAL material "
        f"(same texture, pattern, thickness as described above). Only annotate pieces that are "
        f"100% the same material - do NOT annotate anything that merely looks similar.\n\n"
        f"For each confirmed fascia location call `highlight_fascia` with a tight bbox.\n"
        f"STRICT RULES:\n"
        f"  - Do NOT annotate anything the arrow was not pointing to unless it is the exact same material.\n"
        f"  - Do NOT guess or include nearby elements that differ in texture, thickness, or pattern.\n"
        f"  - Do NOT merge separate pieces into one box.\n"
        f"  - Enclose the complete fascia board/trim - do not cut off the ends.\n\n"
        f"Coordinates: integers, x in [0,{width}], y in [0,{height}]."
    )
    s4 = _call_vlm(step4_prompt, image_bytes, mime_type="image/png",
                   tools=[highlight_fascia], tool_config=fascia_cfg,
                   afc_off=True, label="Phase 1b Step 4", run_id=run_id)

    collected_bboxes = []
    if s4 and s4.function_calls:
        for fc in s4.function_calls:
            if fc.name == "highlight_fascia":
                a = fc.args
                bbox = {k: int(a[k]) for k in ("ymin", "xmin", "ymax", "xmax")}
                collected_bboxes.append(bbox)
                print(f"[Phase 1b] Fascia occurrence found: {bbox}")
                log_response("Phase 1b Fascia BBox", str(bbox))
        print(f"[Phase 1b] Step 4: {len(collected_bboxes)} fascia occurrence(s) found.")
    else:
        print("[Phase 1b] Step 4: No highlight_fascia calls - falling back to tip bbox.")
        collected_bboxes.append(tip_bbox)

    # -- Step 5: Annotate ALL occurrences -------------------------------------
    if not collected_bboxes:
        print("[Phase 1b] No fascia found.")
        return None, [], arrow_info, []

    try:
        with Image.open(annotated_path) as im:
            draw = ImageDraw.Draw(im)
            for b in collected_bboxes:
                draw.rectangle(
                    [b["xmin"], b["ymin"], b["xmax"], b["ymax"]],
                    outline="red", width=5,
                )
            im.save(annotated_path)
        print(f"[Phase 1b] Annotated {len(collected_bboxes)} fascia(s) -> {annotated_path}")
    except Exception as e:
        print(f"[Phase 1b] Error drawing annotation: {e}")
        return None, [], arrow_info, []

    # Single annotated image with all bboxes drawn on the zoom crop (no separate sub-crops)
    multi_crops = [{"occurrence_index": 1, "annotated_path": annotated_path, "bbox": collected_bboxes[0], "description": arrow_info}]

    return annotated_path, collected_bboxes, arrow_info, multi_crops


def run_phase1(pdf_path: str, search_text: str, page_number: int, instance_index: int = 0, run_id: str = None):
    update_heartbeat(run_id)
    result = search_text_in_pdf(pdf_path, search_text, page_number, instance_index=instance_index)
    h_pdf = result.get("full_page_path")
    output = {"full_pdf_part": None, "cropped_image_part": None, "crop_path": None, "full_page_path": h_pdf, "found": result.get("found", False), "_search_crop_path": result.get("crop_path")}
    if not (h_pdf and os.path.exists(h_pdf)): return output
    with open(h_pdf, "rb") as f: output["full_pdf_part"] = types.Part.from_bytes(data=f.read(), mime_type="application/pdf")
    try:
        tr = [v*2 for v in result["text_rect"]] if result.get("text_rect") else None
        dr = [v*2 for v in result["drawing_rect"]] if result.get("drawing_rect") else None
        coords, kn = get_vlm_crop(h_pdf, search_text, text_rect=tr, drawing_rect=dr)
        if coords:
            z_path = f"fascia_zoom_{uuid.uuid4().hex[:6]}_{page_number+1}.png"
            if execute_crop(h_pdf, coords, z_path):
                output["crop_path"], output["keynote_symbol"] = z_path, kn
                output["crop_coords"] = coords
                output["keynote_dimension_label"] = get_keynote_dimension(h_pdf, kn, search_text) if kn else ""
                with open(z_path, "rb") as f: output["cropped_image_part"] = types.Part.from_bytes(data=f.read(), mime_type="image/png")
    except Exception: pass
    if not output["cropped_image_part"] and result.get("crop_path"):
        output["crop_path"] = result["crop_path"]
        with open(result["crop_path"], "rb") as f: output["cropped_image_part"] = types.Part.from_bytes(data=f.read(), mime_type="image/png")
    return output


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Triage / Classification
# ══════════════════════════════════════════════════════════════════════════════

_TRIAGE_PROMPT = """You are a Senior Architectural Technologist analyzing a construction drawing.
Identify the drawing title associated with the highlighted "{keyword}" text, then classify it.
CLASSIFICATION RULES:
- DETAIL / SECTION / PROFILE → GOLD
- ELEVATION → SILVER
- PLAN / RCP / SCHEDULE → BRONZE
CRITICAL: Respond with ONLY a valid JSON object. No explanation, no markdown, no code fences.
Keep "drawing_title" under 10 words.
Output exactly: {{"drawing_title": "<short title>", "authority": "<GOLD|SILVER|BRONZE>"}}"""


def run_phase2(image_part, keyword: str, run_id: str = None):
    update_heartbeat(run_id)
    resp = _gemini_retry(
        MODEL_NAME, [_TRIAGE_PROMPT.format(keyword=keyword), image_part],
        types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
        label="Phase 2", run_id=run_id)
    raw = _get_text(resp)
    if not raw: raise RuntimeError("[PHASE 2 ERROR] Gemini returned empty response.")
    parsed = _parse_json(raw, phase="PHASE 2")
    if "drawing_title" not in parsed or "authority" not in parsed:
        raise RuntimeError(f"[PHASE 2 ERROR] Missing keys.\n\nRaw:\n{raw}")
    parsed["authority"] = parsed["authority"].upper().strip()
    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Dimension Extraction
# ══════════════════════════════════════════════════════════════════════════════

_EXTRACTION_PROMPT = """You are a Senior Architectural Technologist analyzing a construction drawing.
Target component: "<<<KEYWORD>>>". Annotated with RED bounding boxes.

STEP 0 — USE PRE-EXTRACTED LEGEND DIMENSION (HIGHEST PRIORITY):
  If "LEGEND DIMENSION" is provided below, treat as confirmed ground truth → set height_value+unit directly.
  If empty/not found → proceed to Step 1.

STEP 1 — Check diagram dimension labels on/adjacent to RED box. Calculate by subtraction if needed. Null if still nothing.
STEP 2 — Count keynote bubble occurrences in the diagram (small circle/diamond with number/letter). Record as "keynote_occurrences".
STEP 3 — Extract material name and the exact dimension label text read.

RULES: Write inches as 'in', feet as 'ft'. Use null if not determinable.
CRITICAL: Respond ONLY with valid JSON — no explanation, no markdown.
Output: {"height_value": <number|null>, "unit": "<string|null>", "material": "<string|null>", "dimension_label_text": "<string|null>", "keynote_occurrences": <int|null>}"""


def run_phase3(image_part, keyword: str, keynote_symbol: str = "", keynote_dim_label: str = "", run_id: str = None):
    update_heartbeat(run_id)
    prompt = _EXTRACTION_PROMPT.replace("<<<KEYWORD>>>", keyword)
    prompt += ("\n\nLEGEND DIMENSION (pre-read, confirmed): "
               f"{keynote_dim_label!r}\nUse as height_value+unit directly."
               if keynote_dim_label
               else "\n\nLEGEND DIMENSION: not found — use Steps 1-3.")

    resp = _gemini_retry(MODEL_NAME, [prompt, image_part],
                         types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
                         label="Phase 3", run_id=run_id)
    raw = _get_text(resp)
    if not raw: raise RuntimeError("[PHASE 3 ERROR] Gemini returned empty response.")
    return _parse_json(raw, required_keys={"height_value", "unit", "material",
                                           "dimension_label_text", "keynote_occurrences"}, phase="PHASE 3")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Verification
# ══════════════════════════════════════════════════════════════════════════════

_VERIFICATION_PROMPT = """You are a Senior Architectural Reviewer verifying extracted Fascia dimensions.
Verify the component(s) highlighted with a RED bounding box.

CRITICAL CHECKS:
- Does the extracted height match ONLY the region enclosed by the RED box (not a larger overarching dimension)?
- If multiple RED boxes: extracted height MUST be the correct SUM of all box heights.
- Scale consistency, no contradicting notes (SIM, TYP, REFER).
- Dimension clearly attached to Fascia.

Your reason must state whether the dimension matches the top/bottom bounds of the RED box(es).
CRITICAL: Respond ONLY with valid JSON — no explanation, no markdown.
Output: {"verified": <true|false>, "reason": "<justified reason>"}"""


def run_phase4(image_part, extracted_data, keyword: str, run_id: str = None):
    update_heartbeat(run_id)
    resp = _gemini_retry(
        MODEL_NAME,
        [f"EXTRACTED_DATA:\n{json.dumps(extracted_data)}", image_part],
        types.GenerateContentConfig(
            system_instruction=_VERIFICATION_PROMPT, temperature=0.1,
            response_mime_type="application/json",
            response_schema={"type": "object",
                             "properties": {"verified": {"type": "boolean"}, "reason": {"type": "string"}},
                             "required": ["verified", "reason"]},
        ), label="Phase 4", run_id=run_id)
    raw = _get_text(resp)
    if not raw: raise RuntimeError("[PHASE 4 ERROR] Gemini returned empty response.")
    parsed = _parse_json(raw, phase="PHASE 4")
    parsed.setdefault("verified", False); parsed.setdefault("reason", "Unknown")
    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Method Extraction
# ══════════════════════════════════════════════════════════════════════════════

_METHOD_PROMPT = """YOU ARE A MATH EXPERT AND AN EXPERT AT EIFS QUANTITY TAKEOFFS.

Given:
1. IMAGE with target **{keyword}** component(s) in RED bounding boxes.
2. VALIDATOR FINDINGS: {validator_findings}
3. PREVIOUS EXTRACTION: {extractor_response}

YOUR TASK: Find the correct method to calculate the exact width/height of ONLY the RED box regions.

RULES:
- Look at the PHYSICAL extent of the RED boxes vs dimension lines. Do NOT grab the largest nearby dimension if the box is smaller.
- Compare top/bottom edges of RED box to dimension lines. MATCH box bounds to exact dimension lines.
- If no single line matches, use subtraction/addition of adjacent lines.
- Multiple RED boxes: calculate each separately, then SUM them for TOTAL.
- Write feet as 'ft', inches as 'in' (never ' or ") for valid JSON.

OUTPUT — CRITICAL: A SINGLE valid JSON object only. No prose, no markdown.
Schema: {{"method": "<step-by-step method referencing visible dimension values>"}}"""


def run_phase6(image_part, validator_findings, extractor_response, keyword, run_id: str = None):
    update_heartbeat(run_id)
    if image_part is None: raise RuntimeError("[PHASE 6 ERROR] No image provided.")
    prompt = _METHOD_PROMPT.format(
        keyword=keyword,
        validator_findings=json.dumps(validator_findings, indent=1),
        extractor_response=json.dumps(extractor_response, indent=1))

    resp = _gemini_retry(MODEL_NAME, [prompt, image_part],
                         types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
                         label="Phase 6", run_id=run_id)
    raw = _get_text(resp)
    if not raw: raise RuntimeError("[PHASE 6 ERROR] Gemini returned empty response.")
    parsed = _parse_json(raw, phase="PHASE 6")
    if "method" in parsed: return parsed
    first_val = next(iter(parsed.values()), None)
    if first_val: return {"method": str(first_val)}
    raise RuntimeError(f"[PHASE 6 ERROR] No 'method' key in response.\n\nRaw:\n{raw}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — Calculation
# ══════════════════════════════════════════════════════════════════════════════

_CALCULATION_PROMPT = """You are a MATH EXPERT reading construction drawings.

Given:
1. IMAGE with "<<<KEYWORD>>>" component(s) in RED bounding boxes.
2. METHOD to implement: <<<METHOD>>>

YOUR TASK:
  1. Read image — identify every dimension label, elevation marker, and scale note near RED boxes.
  2. Follow METHOD exactly — perform stated arithmetic step by step.
  3. Return final answer as JSON.

RULES:
- DO NOT copy the largest nearby dimension — execute the method's arithmetic.
- If method says subtract elevations, do that exact subtraction.
- Multiple RED boxes: calculate each segment, return SUM as "height".
- Write units as 'ft' and 'in' (not ' or "). Example: 4ft-0in, 7in, 150mm.
- Do NOT invent values. If truly undeterminable → use null.

OUTPUT — CRITICAL: ONLY a valid JSON object. No prose, no markdown.
Required keys: "height" (string with units or null), "width" (string or null), "notes" (one-line arithmetic summary).
For multi-segment: also include "segment_1", "segment_2" etc., with "height" = their SUM."""


def run_phase7(image_part, method_result, keyword: str, run_id: str = None):
    update_heartbeat(run_id)
    if image_part is None: raise RuntimeError("[PHASE 7 ERROR] No image provided.")
    method_text = method_result.get("method", "") if isinstance(method_result, dict) else ""
    if not method_text: method_text = json.dumps(method_result, indent=2)

    prompt = (_CALCULATION_PROMPT
              .replace("<<<KEYWORD>>>", keyword)
              .replace("<<<METHOD>>>",  method_text))

    resp = _gemini_retry(MODEL_NAME, [prompt, image_part],
                         types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
                         label="Phase 7", run_id=run_id)
    raw = _get_text(resp)
    if not raw: raise RuntimeError("[PHASE 7 ERROR] Gemini returned empty response.")
    parsed = _parse_json(raw, phase="PHASE 7")
    parsed = {k.lower(): v for k, v in parsed.items()}
    parsed.setdefault("width", None); parsed.setdefault("height", None); parsed.setdefault("notes", None)
    return parsed


def run_pipeline(pdf_path, search_text, page_number, instance_index: int = 0, run_id: str = None, p1_result=None):
    """Execute phases 1→7 sequentially. Returns status dict.
    If p1_result is provided, Phase 1 is skipped (avoids re-running VLM crop)."""
    import traceback as _tb
    try:
        if p1_result is not None:
            p1 = p1_result
            print(f"--- [Phase 1] '{search_text}' (inst {instance_index}) — using pre-computed result ---")
        else:
            print(f"--- [Phase 1] '{search_text}' (inst {instance_index}) on page {page_number} ---")
            p1 = run_phase1(pdf_path, search_text, page_number, instance_index=instance_index, run_id=run_id)
        if not p1["found"] or not p1["full_pdf_part"]:
            return {"status":"FAILED","error":"Search text not found"}

        full_pdf_part       = p1["full_pdf_part"]
        crop_path           = p1.get("crop_path")
        keynote_symbol      = p1.get("keynote_symbol", "")
        keynote_dim_label   = p1.get("keynote_dimension_label", "")
        
        annotated_path      = None
        fascia_bboxes       = []
        fascia_description  = None
        multi_crops         = []

        if crop_path:
            res1b = run_phase1b(crop_path, search_text, keynote_symbol, run_id=run_id)
            annotated_path, fascia_bboxes, fascia_description, multi_crops = res1b
            if not annotated_path:
                print("--- [Phase 1b] Failed — using original crop for final output.")
                annotated_path = crop_path
        else:
            vlm_input_part = full_pdf_part
            multi_crops = [{"occurrence_index": 1, "annotated_path": None, "bbox": {}, "description": "Full image fallback"}]

        # If no multi_crops were generated (e.g. failure in 1b), create a fallback
        if not multi_crops:
            multi_crops = [{"occurrence_index": 1, "annotated_path": annotated_path or crop_path, "bbox": {}, "description": fascia_description or "Fallback"}]

        occurrence_results = []
        for occ in multi_crops:
            idx = occ["occurrence_index"]
            occ_path = occ["annotated_path"]
            print(f"--- [Pipeline] Processing Fascia Occurrence {idx} ---")
            
            if occ_path and os.path.exists(occ_path):
                with open(occ_path, "rb") as f:
                    vlm_input_part = types.Part.from_bytes(data=f.read(), mime_type="image/png")
            else:
                # Gemini rejects PDF parts with response_mime_type=application/json (400).
                # Rasterize the full page to PNG instead.
                full_page = p1.get("full_page_path", "")
                vlm_input_part = (_pdf_page_to_image_part(full_page) if full_page and os.path.exists(full_page)
                                  else full_pdf_part)

            # --- Phase 2 & 3 in parallel (they are independent) ---
            from concurrent.futures import ThreadPoolExecutor as _TPE
            import traceback as _tb
            print(f"--- [Phase 2+3] Classifying & Extracting Fascia Occ {idx} (parallel) ---")
            def _p2():
                try:    return run_phase2(vlm_input_part, search_text, run_id=run_id)
                except Exception as e: print(f"[ERROR] Phase 2 failed for Occ {idx}: {e}"); return {"error": str(e)}
            def _p3():
                try:    return run_phase3(vlm_input_part, search_text, keynote_symbol, keynote_dim_label, run_id=run_id)
                except Exception as e: print(f"[ERROR] Phase 3 failed for Occ {idx}: {e}"); return {"error": str(e)}
            with _TPE(max_workers=2) as _ex:
                f2, f3 = _ex.submit(_p2), _ex.submit(_p3)
                triage_result, extraction_result = f2.result(), f3.result()

            # --- Phases 4, 6, 7 are removed for performance (not used by exporter) ---
            occurrence_results.append({
                "occurrence_index": idx,
                "phase2": triage_result,
                "phase3": extraction_result,
                "annotated_path": occ_path
            })

        # Return the first occurrence as the primary result for backward compatibility
        primary = occurrence_results[0]
        return {
            "status": "SUCCESS",
            "keynote_symbol": keynote_symbol,
            "keynote_dimension_label": keynote_dim_label,
            "phase1b":  {"bboxes": fascia_bboxes, "description": fascia_description, "multi_crops": multi_crops},
            "occurrence_results": occurrence_results,
            # Backward compatibility fields
            "phase2":   primary["phase2"],
            "phase3":   primary["phase3"],
            "crop_path": crop_path,
            "crop_coords": p1.get("crop_coords"),
            "annotated_path": annotated_path,
            "full_page_path": p1.get("full_page_path"),
        }

    except Exception as e:
        print(f"[CRITICAL ERROR] Fascia pipeline failed: {e}")
        import traceback; traceback.print_exc()
        return {"status": "CRITICAL_ERROR", "error": str(e)}

def _rects_overlap(r1, r2, threshold=0.5):
    """Check if two crop-coord dicts overlap significantly (intersection / smaller area > threshold)."""
    if not r1 or not r2:
        return False
    ix0 = max(r1["xmin"], r2["xmin"]); iy0 = max(r1["ymin"], r2["ymin"])
    ix1 = min(r1["xmax"], r2["xmax"]); iy1 = min(r1["ymax"], r2["ymax"])
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    a1 = max(1, (r1["xmax"] - r1["xmin"]) * (r1["ymax"] - r1["ymin"]))
    a2 = max(1, (r2["xmax"] - r2["xmin"]) * (r2["ymax"] - r2["ymin"]))
    return inter / min(a1, a2) > threshold


def run_full_document(pdf_path: str, run_id: str):
    import queue, threading

    search_keywords = ["Fascia", "FASCIA"]
    _WORKERS = 3

    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    print(f"--- [Fascia Pipeline] Starting: {num_pages} pages, {_WORKERS} worker threads.")

    # Shared page queue: every page index goes in, threads pull one at a time
    page_queue = queue.Queue()
    for i in range(num_pages):
        page_queue.put(i)

    # Thread-safe collectors
    results_lock = threading.Lock()
    results = []
    ann_imgs = []
    all_temp = []

    def _worker(thread_id):
        while True:
            # Check if pipeline has been cancelled
            if is_cancelled():
                print(f"--- [Fascia T{thread_id}] Cancelled — stopping.")
                return

            try:
                page_idx = page_queue.get_nowait()
            except queue.Empty:
                return  # no more pages

            try:
                # Quick scan: does this page have the keyword?
                doc_local = fitz.open(pdf_path)
                page = doc_local[page_idx]
                has_match = False
                for kw in search_keywords:
                    if page.search_for(kw):
                        has_match = True
                        break
                doc_local.close()

                if not has_match:
                    page_queue.task_done()
                    continue  # no keyword on this page → grab next page immediately

                print(f"--- [Fascia T{thread_id}] Page {page_idx+1} — keyword found, processing...")

                # Process all instances on this page
                seen = set()
                seen_crops = []
                doc_local = fitz.open(pdf_path)
                page = doc_local[page_idx]

                for kw in search_keywords:
                    insts = page.search_for(kw)
                    for idx in range(len(insts)):
                        if idx in seen:
                            continue
                        seen.add(idx)
                        update_heartbeat(run_id)
                        try:
                            print(f"--- [Fascia T{thread_id}] Page {page_idx+1} inst {idx} '{kw}' — Phase 1 ---")
                            p1 = run_phase1(pdf_path, kw, page_idx, instance_index=idx, run_id=run_id)

                            t_files = []
                            if p1.get("full_page_path"): t_files.append(p1["full_page_path"])
                            if p1.get("crop_path"):      t_files.append(p1["crop_path"])
                            if p1.get("_search_crop_path"): t_files.append(p1["_search_crop_path"])

                            if not p1.get("found") or not p1.get("full_pdf_part"):
                                print(f"--- [Fascia T{thread_id}] Page {page_idx+1} inst {idx} — not found, skipping.")
                                with results_lock:
                                    all_temp.extend(t_files)
                                continue

                            crop_coords = p1.get("crop_coords")
                            if crop_coords and any(_rects_overlap(crop_coords, prev) for prev in seen_crops):
                                print(f"--- [Fascia T{thread_id}] Page {page_idx+1} inst {idx} — dedup, skipping.")
                                with results_lock:
                                    all_temp.extend(t_files)
                                continue
                            if crop_coords:
                                seen_crops.append(crop_coords)

                            res = run_pipeline(pdf_path, kw, page_idx, instance_index=idx, run_id=run_id, p1_result=p1)

                            m_crops = res.get("phase1b", {}).get("multi_crops", [])
                            a_imgs = []
                            if m_crops:
                                for mc in m_crops:
                                    if mc.get("annotated_path") and os.path.exists(mc["annotated_path"]):
                                        a_imgs.append(mc["annotated_path"])
                            elif res.get("annotated_path") and os.path.exists(res["annotated_path"]):
                                a_imgs.append(res["annotated_path"])

                            with results_lock:
                                results.append({"page": page_idx, "keyword": kw, "result": res})
                                ann_imgs.extend(a_imgs)
                                all_temp.extend(t_files)

                        except Exception as e:
                            import traceback
                            print(f"--- [Fascia T{thread_id}] Page {page_idx+1} inst {idx} '{kw}' failed: {e}")
                            traceback.print_exc()

                doc_local.close()
                page_queue.task_done()

            except Exception as e:
                import traceback
                print(f"--- [Fascia T{thread_id}] Page {page_idx+1} error: {e}")
                traceback.print_exc()
                page_queue.task_done()

    # Launch worker threads
    threads = []
    for tid in range(_WORKERS):
        t = threading.Thread(target=_worker, args=(tid+1,))
        t.start()
        threads.append(t)

    # Wait for all worker threads to finish (30-min safety timeout)
    for t in threads:
        t.join(timeout=1800)

    # Drain any remaining items if threads timed out
    while not page_queue.empty():
        try:
            page_queue.get_nowait()
            page_queue.task_done()
        except queue.Empty:
            break

    # Sort results by page number
    results.sort(key=lambda r: r["page"])

    print(f"--- [Fascia Pipeline] Done. {len(ann_imgs)} annotated images from {len(results)} results.")

    # Build annotated_entries: track which page each annotated image came from
    # so they can be embedded into the debug PDF with sub-page labels (10a, 10b…)
    # ann_imgs is filled in order as results are processed; match them to page_idx via results.
    annotated_entries = []
    # Use results (sorted by page) to correlate annotated images with page indices
    img_iter = iter(ann_imgs)
    page_sub_counter = {}  # page_idx -> next sub-index letter offset
    for r in results:
        page_idx = r["page"]
        res = r.get("result", {})
        m_crops = res.get("phase1b", {}).get("multi_crops", [])
        # Count how many annotated images came from this result
        n_imgs = len([mc for mc in m_crops if mc.get("annotated_path") and os.path.exists(mc["annotated_path"])]) if m_crops else (1 if res.get("annotated_path") and os.path.exists(res.get("annotated_path", "")) else 0)
        for _ in range(n_imgs):
            try:
                img_path = next(img_iter)
                sub_offset = page_sub_counter.get(page_idx, 0)
                page_sub_counter[page_idx] = sub_offset + 1
                annotated_entries.append({
                    "page_idx": page_idx,
                    "img_path": img_path,
                    "source": "Fascia",
                    "sub_idx": sub_offset,
                })
            except StopIteration:
                break

    for f in set(all_temp):
        try:
            if os.path.exists(f): os.remove(f)
        except Exception: pass
    return {"status": "SUCCESS" if ann_imgs else "NO_MATCHES", "annotated_entries": annotated_entries, "page_results": results}

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        print(json.dumps(run_pipeline(sys.argv[1], sys.argv[2], int(sys.argv[3])), indent=2))
