"""
Reveal_Gemini.py — Full single-file pipeline (phases 0-7 + SearchText + tools).
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
LOG_FILE = os.path.join(os.path.dirname(__file__), "pipeline.log")
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
    """Call client.models.generate_content with exponential backoff on 429/503."""
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
    """Safely extract text from a Gemini response (handles None .text)."""
    if not response:
        return None
    if response.text:
        return response.text
    try:
        parts = [p.text for c in (response.candidates or [])
                 for p in (c.content.parts or []) if getattr(p, "text", None)]
        return " ".join(parts) if parts else None
    except Exception:
        return None


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
    """Close a truncated JSON object (matches original per-phase repair logic)."""
    text = text.rstrip()
    if text.startswith("{") and not text.endswith("}"):
        if text.count('"') % 2 == 1:   # inside an unclosed string — close it
            text += '"'
        text = text.rstrip(",").rstrip() + "}"
    return text


def _parse_json(raw, required_keys=None, phase=""):
    """Strip fences, attempt JSON parse + repair, fill missing keys with None."""
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
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"[{phase} ERROR] Invalid JSON.\n\nRaw:\n{raw}")


# Thin model wrapper kept for backward compat
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
    doc  = fitz.open(file_path)
    page = doc[page_number]
    all_drawings    = page.get_drawings()
    text_instances  = page.search_for(search_text)
    print(f"Total drawings on page {page_number}: {len(all_drawings)}")

    result = {"found": False, "text_rect": None, "drawing_rect": None,
              "crop_path": None, "full_page_path": None}

    if not text_instances or instance_index >= len(text_instances):
        print(f"'{search_text}' (instance {instance_index}) not found on page {page_number}.")
        doc.close(); return result

    result["found"] = True
    inst = text_instances[instance_index]
    result["text_rect"] = [inst.x0, inst.y0, inst.x1, inst.y1]
    print(f"Found '{search_text}' (instance {instance_index}) at: {inst}")

    sh = page.new_shape(); sh.draw_rect(inst); sh.finish(color=(1,0,0), width=2); sh.commit()

    for drawing in all_drawings:
        r = drawing["rect"]
        if inst.x0 < r[2] and inst.x1 > r[0] and inst.y0 < r[3] and inst.y1 > r[1]:
            print(f"Drawing near text: {drawing['type']} at {r}")
            # Refined: We no longer draw heuristic boxes here as they are often incorrect.
            # We just track the rect to inform the VLM zoom window.
            result["drawing_rect"] = list(r)
            break

    img          = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    full_pdf_path = f"reveal_extracted_{uuid.uuid4().hex[:6]}_{page_number+1}.pdf"
    out = fitz.open()
    pg  = out.new_page(width=img.width, height=img.height)
    pg.insert_image(pg.rect, pixmap=img)
    out.save(full_pdf_path); out.close()
    result["full_page_path"] = full_pdf_path
    print(f"Saved full page to {full_pdf_path}")
    doc.close()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 TOOL — VLM Zoom
# ══════════════════════════════════════════════════════════════════════════════

_MAX_EDGE = 3000


def crop_image(ymin: int, xmin: int, ymax: int, xmax: int, keynote_symbol: str = ""):
    """
    Call with the bounding box of the diagram/detail in the image shown.
    Args:
        ymin: Top Y.  xmin: Left X.  ymax: Bottom Y.  xmax: Right X.
        keynote_symbol: Keynote number/symbol if matched via notes list, else "".
    """
    return "Cropping image..."


def _render_region_to_jpeg(pdf_path: str, region: fitz.Rect):
    """Render region of first page to JPEG. Returns (jpeg_bytes, w, h, scale)."""
    doc  = fitz.open(pdf_path)
    page = doc[0]
    region = region & page.rect
    scale  = min(_MAX_EDGE / max(region.width, region.height), 3.0)
    pix    = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=region)
    data   = pix.tobytes("jpeg")
    w, h   = pix.width, pix.height
    doc.close()
    print(f"[VLM Zoom] PDF {int(region.width)}×{int(region.height)}pt → JPEG {w}×{h}px scale={scale:.3f}")
    return data, w, h, scale


def get_vlm_crop(pdf_path: str, search_text: str, text_rect=None, drawing_rect=None):
    """Returns (coords_dict, keynote_symbol) or (None, '') on failure."""
    doc  = fitz.open(pdf_path)
    page = doc[0]
    window = page.rect
    win_x0, win_y0 = window.x0, window.y0
    print(f"[VLM Zoom] Full page: ({int(win_x0)},{int(win_y0)})→({int(window.x1)},{int(window.y1)})")

    try:
        jpeg_bytes, img_w, img_h, sub_scale = _render_region_to_jpeg(pdf_path, window)
    except Exception as e:
        print(f"[VLM Zoom] Render failed: {e}"); doc.close(); return None, ""
    doc.close()

    def to_sub(r):
        return (int((r[0]-win_x0)*sub_scale), int((r[1]-win_y0)*sub_scale),
                int((r[2]-win_x0)*sub_scale), int((r[3]-win_y0)*sub_scale))

    text_hint = ""
    if text_rect:
        tx0, ty0, tx1, ty1 = to_sub(text_rect)
        text_hint = (f" The label '{search_text}' was found at approx "
                     f"x=[{tx0}–{tx1}], y=[{ty0}–{ty1}].")

    prompt = (
        f"This image ({img_w}×{img_h} px) is a full architectural construction sheet.{text_hint}\n\n"
        f"TASK: Find the boundary of the ENTIRE architectural detail drawing block associated with '{search_text}'.\n\n"
        f"STRATEGY:\n"
        f"1. Look at the location of '{search_text}' indicated above.\n"
        f"2. Decide: is it inside a 'Notes/Keynotes/Legend' list — or directly inside a diagram?\n"
        f"3. IF in Notes/Keynotes/Legend:\n"
        f"   a. READ EACH ROW carefully: [number] [description text].\n"
        f"   b. Find the ONE row whose description literally contains '{search_text}' (case-insensitive, exact word).\n"
        f"   c. Record that row's number/symbol — that is your keynote to find in diagrams.\n"
        f"4. Scan the ENTIRE SHEET for the diagram containing that keynote bubble.\n"
        f"5. IF '{search_text}' is already directly in a diagram, focus on that diagram.\n\n"
        f"HOW TO CROP — STRICT BOUNDARY RULES:\n"
        f"- You MUST capture the COMPLETE diagram. NEVER clip any part of it.\n"
        f"- ymin: Set 30-50 px ABOVE the topmost element (dimension lines, extension lines, keynote bubbles, or top of the drawing frame).\n"
        f"- ymax: Set 30-50 px BELOW the Drawing Title/SCALE text at the bottom of the detail block.\n"
        f"- xmin: Set 30-50 px LEFT of the leftmost element (outermost dimension line, callout, or drawing frame edge).\n"
        f"- xmax: Set 30-50 px RIGHT of the rightmost element (outermost dimension line, callout, or drawing frame edge).\n"
        f"- Include ALL: dimension lines, extension lines, keynote bubbles, leader lines, section cut marks, \n"
        f"  hatching/patterns, material annotations, elevation markers, and the detail title.\n"
        f"- When in doubt, crop LARGER rather than smaller — it is far worse to clip part of the \n"
        f"  diagram than to include a small amount of surrounding whitespace.\n\n"
        f"CRITICAL RULES:\n"
        f"- ONE bounding box only.\n"
        f"- If from Notes list: crop MUST NOT include the Notes section — crop the DIAGRAM the keynote points to.\n"
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
                fc   = response.function_calls[0]
                args = fc.args
                sy0, sx0, sy1, sx1 = int(args["ymin"]), int(args["xmin"]), int(args["ymax"]), int(args["xmax"])
                keynote_symbol = args.get("keynote_symbol", "")
                coords = {
                    "ymin": int(win_y0 + sy0/sub_scale), "xmin": int(win_x0 + sx0/sub_scale),
                    "ymax": int(win_y0 + sy1/sub_scale), "xmax": int(win_x0 + sx1/sub_scale),
                }
                print(f"[VLM Zoom] PDF coords: {coords}  keynote='{keynote_symbol}'")
                # When keynote is found, refine crop to anchor on the keynote bubble in diagrams
                if keynote_symbol:
                    refined = _refine_keynote_crop(
                        keynote_symbol, search_text, jpeg_bytes,
                        img_w, img_h, sub_scale, win_x0, win_y0, text_rect)
                    if refined:
                        coords = refined
                        print(f"[VLM Zoom] Refined keynote crop → {coords}")
                return coords, keynote_symbol
            else:
                print(f"[VLM Zoom] No function call. reason={response.candidates[0].finish_reason if response.candidates else 'unknown'}")
                return None, ""
        except Exception as e:
            if _is_rate_limit(e) and attempt < 3:
                wait = min(10 * (2 ** (attempt-1)), _MAX_RETRY_WAIT)
                print(f"[VLM Zoom] Rate limit attempt {attempt}/3 — retry in {wait}s")
                _heartbeat_sleep(wait)
            else:
                print(f"[VLM Zoom Error] {e}"); return None, ""


def _refine_keynote_crop(keynote_symbol, search_text, jpeg_bytes,
                        img_w, img_h, sub_scale, win_x0, win_y0,
                        text_rect=None):
    """
    Second-pass VLM crop: specifically finds where keynote bubble
    '{keynote_symbol}' physically appears INSIDE A DIAGRAM (not in the
    Notes/Legend list) and crops that exact diagram.
    Returns refined coords dict or None on failure.
    """
    avoid_hint = ""
    if text_rect:
        tx0 = int((text_rect[0] - win_x0) * sub_scale)
        ty0 = int((text_rect[1] - win_y0) * sub_scale)
        tx1 = int((text_rect[2] - win_x0) * sub_scale)
        ty1 = int((text_rect[3] - win_y0) * sub_scale)
        avoid_hint = (
            f"\nIMPORTANT: The keyword '{search_text}' was found in the Notes/Legend area "
            f"near pixel x=[{tx0}–{tx1}], y=[{ty0}–{ty1}]. "
            f"Do NOT crop that Notes/Legend area — you must crop the DIAGRAM that "
            f"contains the keynote bubble instead."
        )

    prompt = (
        f"This image ({img_w}×{img_h} px) is a full architectural construction sheet."
        f"{avoid_hint}\n\n"
        f"TASK: Locate the keynote bubble/circle/diamond labeled '{keynote_symbol}' that appears "
        f"INSIDE A DIAGRAM on this sheet — NOT in the Notes/Legend/Keynotes list section.\n\n"
        f"The keynote '{keynote_symbol}' refers to '{search_text}'. "
        f"It appears as a small circle, diamond, or numbered marker with '{keynote_symbol}' "
        f"written inside it, usually connected to a part of a diagram by a leader line/arrow.\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Scan ALL diagram/detail blocks on this sheet.\n"
        f"2. Find the diagram that contains a keynote bubble labeled '{keynote_symbol}'.\n"
        f"3. Crop the COMPLETE diagram that contains this keynote bubble.\n\n"
        f"CROP RULES:\n"
        f"- The keynote bubble '{keynote_symbol}' MUST be visible inside your crop area.\n"
        f"- Include the full diagram: title, scale text, dimension lines, callouts, hatching.\n"
        f"- Add 30-50 px margin on all sides.\n"
        f"- Do NOT crop the Notes/Legend/Keynotes list section.\n"
        f"- If keynote '{keynote_symbol}' appears in MULTIPLE diagrams, crop the one where "
        f"it is most prominent or has the clearest detail.\n\n"
        f"CRITICAL:\n"
        f"- ONE bounding box only.\n"
        f"- The crop MUST contain the keynote bubble '{keynote_symbol}' inside it.\n"
        f"- Pass '{keynote_symbol}' as keynote_symbol.\n"
        f"- Coordinates: integers, x∈[0,{img_w}], y∈[0,{img_h}].\n"
        f"- Crop LARGER rather than smaller to capture the complete diagram."
    )

    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=_ZOOM_MODEL,
                contents=[prompt, types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    tools=[crop_image],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY", allowed_function_names=["crop_image"]
                        )
                    ),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            if response.function_calls:
                fc = response.function_calls[0]
                args = fc.args
                sy0 = int(args["ymin"]); sx0 = int(args["xmin"])
                sy1 = int(args["ymax"]); sx1 = int(args["xmax"])
                refined_coords = {
                    "ymin": int(win_y0 + sy0 / sub_scale),
                    "xmin": int(win_x0 + sx0 / sub_scale),
                    "ymax": int(win_y0 + sy1 / sub_scale),
                    "xmax": int(win_x0 + sx1 / sub_scale),
                }
                print(f"[VLM Keynote Refine] Refined PDF coords: {refined_coords}")
                return refined_coords
            else:
                print("[VLM Keynote Refine] No function call returned.")
                return None
        except Exception as e:
            if _is_rate_limit(e) and attempt < 3:
                wait = min(10 * (2 ** (attempt - 1)), _MAX_RETRY_WAIT)
                print(f"[VLM Keynote Refine] Rate limit — retry in {wait}s")
                _heartbeat_sleep(wait)
            else:
                print(f"[VLM Keynote Refine] Error: {e}")
                return None
    return None


_CROP_PAD = 15   # extra PDF-pt padding around VLM coords to avoid clipping edges
_MIN_CROP = 120  # minimum crop dimension in PDF points


def execute_crop(pdf_path: str, coords: dict, output_path: str) -> bool:
    """Crop PDF page at coords and save as PNG (2× scale).
    Adds safety padding and enforces a minimum crop size so the
    complete diagram/component is always captured."""
    doc  = fitz.open(pdf_path)
    page = doc[0]
    pw, ph = page.rect.width, page.rect.height

    # Apply safety padding
    x0 = max(0,  coords["xmin"] - _CROP_PAD)
    y0 = max(0,  coords["ymin"] - _CROP_PAD)
    x1 = min(pw, coords["xmax"] + _CROP_PAD)
    y1 = min(ph, coords["ymax"] + _CROP_PAD)

    # Enforce minimum crop size — expand symmetrically if too small
    cw, ch = x1 - x0, y1 - y0
    if cw < _MIN_CROP:
        cx = (x0 + x1) / 2
        x0 = max(0,  cx - _MIN_CROP / 2)
        x1 = min(pw, cx + _MIN_CROP / 2)
    if ch < _MIN_CROP:
        cy = (y0 + y1) / 2
        y0 = max(0,  cy - _MIN_CROP / 2)
        y1 = min(ph, cy + _MIN_CROP / 2)

    if x1 <= x0 or y1 <= y0:
        print(f"[VLM Zoom] Invalid crop rect: ({x0},{y0})→({x1},{y1})"); doc.close(); return False

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=fitz.Rect(x0, y0, x1, y1))
    pix.save(output_path); doc.close()
    print(f"[VLM Zoom] Crop saved → {output_path} ({pix.width}×{pix.height}px)")
    return True


def get_keynote_dimension(pdf_path: str, keynote_symbol: str, search_text: str) -> str:
    """Find dimension value in the Notes/Legend row for keynote_symbol. Returns '' if none."""
    if not keynote_symbol:
        return ""
    try:
        jpeg_bytes, img_w, img_h, _ = _render_region_to_jpeg(pdf_path, fitz.open(pdf_path)[0].rect)
    except Exception as e:
        print(f"[get_keynote_dimension] Render failed: {e}"); return ""

    prompt = (
        f"Full architectural sheet ({img_w}×{img_h} px).\n"
        f"Find the Notes/Keynotes/Legend table. Locate row '{keynote_symbol}' (describes '{search_text}').\n"
        f"Does that row contain a dimension (e.g. '3/4\"', '2in', '150mm')?\n"
        f"If YES — return ONLY the dimension value. If NO — return the single word: NONE"
    )
    try:
        resp = client.models.generate_content(
            model=_ZOOM_MODEL,
            contents=[prompt, types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        raw = (resp.text or "").strip()
        print(f"[get_keynote_dimension] keynote '{keynote_symbol}' → {raw!r}")
        return "" if raw.upper() == "NONE" or not raw else raw
    except Exception as e:
        print(f"[get_keynote_dimension] VLM failed: {e}"); return ""


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1b — Component Highlighting
# ══════════════════════════════════════════════════════════════════════════════

_REVEAL_MODEL = "gemini-3-pro-preview"
_ZOOM_PAD     = 120


def locate_arrow_tip(ymin: int, xmin: int, ymax: int, xmax: int):
    """
    Call this tool with a small tight bbox around the exact pixel location
    where the leader line ENDS (the arrowhead tip in the drawing).

    Args:
        ymin: Top pixel of the arrowhead region.
        xmin: Left pixel of the arrowhead region.
        ymax: Bottom pixel of the arrowhead region.
        xmax: Right pixel of the arrowhead region.
    """
    return "Arrow tip location received."


def highlight_component(ymin: int, xmin: int, ymax: int, xmax: int):
    """
    Call this tool once for EACH separate location where the component material
    is present in the image shown to you.

    Args:
        ymin: Top pixel of one component occurrence.
        xmin: Left pixel of one component occurrence.
        ymax: Bottom pixel of one component occurrence.
        xmax: Right pixel of one component occurrence.
    """
    return "Component location recorded."


def locate_all_occurrences(ymin: int, xmin: int, ymax: int, xmax: int):
    """
    Call this tool ONCE PER OCCURRENCE of the keyword found in the image.
    Box tightly around the keyword label text AND its directly associated diagram area.

    Args:
        ymin: Top pixel of one keyword occurrence.
        xmin: Left pixel of one keyword occurrence.
        ymax: Bottom pixel of one keyword occurrence.
        xmax: Right pixel of one keyword occurrence.
    """
    return "Occurrence location recorded."


def _call_vlm(prompt, image_bytes, mime_type="image/png", tools=None,
              tool_config=None, afc_off=False, label="Phase 1b"):
    """Gemini call with retry for Phase 1b. Returns response or None."""
    cfg = dict(temperature=0.0)
    if tools:      cfg["tools"]      = tools
    if tool_config: cfg["tool_config"] = tool_config
    if afc_off:    cfg["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)

    for attempt in range(1, 4):
        try:
            return client.models.generate_content(
                model=_REVEAL_MODEL,
                contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                config=types.GenerateContentConfig(**cfg),
            )
        except Exception as e:
            if _is_rate_limit(e) and attempt < 3:
                w = min(10 * (2**(attempt-1)), _MAX_RETRY_WAIT)
                print(f"[{label}] Rate limit — retry in {w}s")
                _heartbeat_sleep(w)
            else:
                print(f"[{label}] Error: {e}"); return None
    return None


def _zoom_region(image_bytes, bbox, pad=_ZOOM_PAD):
    """Crop bbox+pad from image_bytes. Returns (jpeg_bytes, off_x, off_y, w, h)."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        W, H = img.size
        x0, y0 = max(0, bbox["xmin"]-pad), max(0, bbox["ymin"]-pad)
        x1, y1 = min(W, bbox["xmax"]+pad), min(H, bbox["ymax"]+pad)
        crop = img.crop((x0, y0, x1, y1))
        buf  = io.BytesIO(); crop.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), x0, y0, crop.width, crop.height


def _crop_and_annotate_occurrence(image_bytes, bbox, base_path, idx, color="red", pad=80):
    """Crop image around bbox+pad, draw bounding box, save PNG. Returns path."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        W, H = img.size
        x0, y0 = max(0, bbox["xmin"]-pad), max(0, bbox["ymin"]-pad)
        x1, y1 = min(W, bbox["xmax"]+pad), min(H, bbox["ymax"]+pad)
        crop = img.crop((x0, y0, x1, y1)).copy()
    draw = ImageDraw.Draw(crop)
    draw.rectangle([bbox["xmin"]-x0, bbox["ymin"]-y0, bbox["xmax"]-x0, bbox["ymax"]-y0],
                   outline=color, width=5)
    out = f"{base_path}_occurrence_{idx}.png"
    crop.save(out, format="PNG")
    print(f"[Phase 1b] Saved occurrence {idx} → {out}")
    return out


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


def _draw_boxes_on_image(annotated_path, bboxes, img_w, img_h):
    """Draw red bounding boxes on the annotated image file in-place."""
    with Image.open(annotated_path) as im:
        draw = ImageDraw.Draw(im)
        for b in bboxes:
            draw.rectangle(_expand_box(b, img_w, img_h), outline="red", width=5)
        im.save(annotated_path)


def _fc_to_bbox(fc_args):
    return {k: int(fc_args[k]) for k in ("ymin","xmin","ymax","xmax")}


def _expand_box(b, img_w, img_h, pad=0):
    """Utility to convert a bbox dict to PIL-friendly [x0, y0, x1, y1]."""
    return [max(0, b["xmin"]-pad), max(0, b["ymin"]-pad), min(img_w, b["xmax"]+pad), min(img_h, b["ymax"]+pad)]

def run_phase1b(image_path: str, keyword: str, keynote_symbol: str = "", run_id: str = None):
    """
    PHASE 1b: Component Highlighting (Intermediate Step)
    Goal: Identify the '{keyword}' component in the cropped detail image and highlight it
          with a red bounding box.

    Two flows depending on whether a keynote bubble was detected:

      KEYNOTE FLOW (keynote_symbol is set):
        Step 1 — Arrow Info    : VLM describes where the keynote arrow ends.
        Step 2a— Bubble Locate : Find the keynote bubble on full image → zoom in.
        Step 2b— Tip Trace     : Trace arrow to its exact tip inside the zoomed bubble view.
        Step 3 — Material Desc : Verify what the arrow tip is touching.
        Step 4 — Annotate      : VLM boxes the arrow touch point on the zoom → map to full image.
        Step 5 — Draw          : Draw red box → return.

      NO-KEYNOTE FLOW (keynote_symbol is empty):
        Step 0 — Count         : VLM counts ALL occurrences of keyword and returns a bbox per one.
        Step A — Locate Keyword: VLM finds the red-highlighted keyword text in the crop.
        Step B — Box Component : VLM boxes the element the keyword is labelling + surrounding diagram.
        Step C — Draw          : Draw red box → return.

        If keyword appears MORE THAN ONCE:
          Each occurrence is CROPPED separately and annotated independently.
          Returns multi_crops: list of {annotated_path, bbox, description}.
    """
    update_heartbeat(run_id)
    if not image_path or not os.path.exists(image_path):
        print(f"[Phase 1b] Image not found: {image_path}"); return None, [], "", []

    with open(image_path, "rb") as f: image_bytes = f.read()
    with Image.open(io.BytesIO(image_bytes)) as img: width, height = img.size
    print(f"[Phase 1b] {image_path} ({width}×{height}px)")

    base, ext = os.path.splitext(image_path)
    annotated_path = f"{base}_annotated{ext or '.png'}"
    shutil.copy2(image_path, annotated_path)

    reveal_cfg = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
        mode="ANY", allowed_function_names=["highlight_component"]))
    tip_cfg = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
        mode="ANY", allowed_function_names=["locate_arrow_tip"]))

    # ── NO-KEYNOTE FLOW ──────────────────────────────────────────────────────
    if not keynote_symbol:
        print("[Phase 1b] No keynote — keyword-text flow (Step 0 Count).")

        # ── Step 0: Count ALL occurrences ─────────────────────────────────────
        all_occurrences_cfg = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=["locate_all_occurrences"]
            )
        )
        step0_prompt = (
            f"This is a cropped architectural drawing ({width}×{height} px).\n\n"
            f"The text '{keyword}' may appear ONE or MORE times in this image "
            f"(highlighted with a RED or YELLOW background/box).\n\n"
            f"TASK:\n"
            f"  For EVERY separate occurrence of the '{keyword}' label you can see:\n"
            f"  - Call `locate_all_occurrences` once per occurrence.\n"
            f"  - Provide a bounding box that covers the '{keyword}' label TEXT "
            f"    AND its directly associated detail diagram / element.\n"
            f"  - Make the box large enough to include the full drawing associated with that instance.\n\n"
            f"RULES:\n"
            f"  - If '{keyword}' appears 3 times → call the tool 3 times.\n"
            f"  - Do NOT merge multiple occurrences into one box.\n"
            f"  - Do NOT annotate unrelated elements.\n\n"
            f"Coordinates: integers, x∈[0,{width}], y∈[0,{height}]."
        )

        s0 = _call_vlm(step0_prompt, image_bytes, mime_type="image/png",
                       tools=[locate_all_occurrences], tool_config=all_occurrences_cfg,
                       afc_off=True, label="P1b-Step0")

        occurrence_bboxes = []
        if s0 and s0.function_calls:
            for fc in s0.function_calls:
                if fc.name == "locate_all_occurrences":
                    occurrence_bboxes.append(_fc_to_bbox(fc.args))
        
        num_occurrences = len(occurrence_bboxes)
        print(f"[Phase 1b] Detected {num_occurrences} occurrence(s) of '{keyword}'.")
        description = f"No keynote found — isolated '{keyword}' diagram detected {num_occurrences} time(s)."

        # If zero occurrences, fallback
        if not occurrence_bboxes:
            collected_bboxes = [{"ymin":0,"xmin":0,"ymax":height,"xmax":width}]
        else:
            collected_bboxes = occurrence_bboxes

        # ── Step C: Draw ──────────────────────────────────────────────────────
        print(f"[Phase 1b] Drawing annotation(s) → {annotated_path}")
        try:
            with Image.open(annotated_path) as im:
                draw = ImageDraw.Draw(im)
                for b in collected_bboxes:
                    draw.rectangle(_expand_box(b, width, height), outline="red", width=5)
                im.save(annotated_path)
        except Exception as e: print(f"Annotation error: {e}")

        # If multiple occurrences, we just return the crop with all of them annotated.
        # But we could also crop each one individually if `run_pipeline` expects it.
        # `multi_crops` handles this.
        multi_crops = []
        for idx, b in enumerate(collected_bboxes, 1):
            if len(collected_bboxes) > 1:
                crop_path = _crop_and_annotate_occurrence(image_bytes, b, base, idx)
                multi_crops.append({"occurrence_index": idx, "annotated_path": crop_path, "bbox": b, "description": description})
            else:
                multi_crops.append({"occurrence_index": 1, "annotated_path": annotated_path, "bbox": b, "description": description})

        return annotated_path, collected_bboxes, description, multi_crops

    # ── KEYNOTE FLOW ─────────────────────────────────────────────────────────
    print(f"[Phase 1b] Keynote Flow for '{keynote_symbol}'.")
    bubble_label = f"keynote bubble '{keynote_symbol}'"

    # Step 1: Arrow Info (Merged out to just use bubble trace)
    # Removing this step saves 1 LLM call. We can directly trace the tip.

    # Step 2a: Bubble Locate
    print("[Phase 1b] Step 2a: Bubble Locate.")
    step2a_prompt = (
        f"This is a construction detail drawing ({width}×{height} px).\n\n"
        f"TASK: Find the {bubble_label}.\n"
        f"Call `locate_arrow_tip` with a bbox tightly enclosing JUST the bubble circle/number.\n"
        f"Coordinates: integers, x∈[0,{width}], y∈[0,{height}]."
    )
    s2a = _call_vlm(step2a_prompt, image_bytes, mime_type="image/png", tools=[locate_arrow_tip], tool_config=tip_cfg, afc_off=True, label="P1b-Step2a")
    bubble_bbox = _fc_to_bbox(s2a.function_calls[0].args) if s2a and s2a.function_calls else {"ymin":0,"xmin":0,"ymax":height,"xmax":width}

    _BUBBLE_PAD = 350
    try: bubble_zoom_bytes, bub_off_x, bub_off_y, bub_w, bub_h = _zoom_region(image_bytes, bubble_bbox, _BUBBLE_PAD)
    except Exception: bubble_zoom_bytes, bub_off_x, bub_off_y, bub_w, bub_h = image_bytes, 0, 0, width, height

    # Step 2b: Tip Trace
    print("[Phase 1b] Step 2b: Tip Trace.")
    step2b_prompt = (
        f"This is a zoomed region ({bub_w}×{bub_h} px) centred on the {bubble_label}.\n\n"
        f"TASK: Follow the leader line from the keynote bubble to where it ENDS.\n"
        f"Call `locate_arrow_tip` with a tight bbox around the ARROWHEAD TIP ONLY.\n"
        f"Coordinates: integers, x∈[0,{bub_w}], y∈[0,{bub_h}]."
    )
    s2b = _call_vlm(step2b_prompt, bubble_zoom_bytes, mime_type="image/jpeg", tools=[locate_arrow_tip], tool_config=tip_cfg, afc_off=True, label="P1b-Step2b")
    if s2b and s2b.function_calls:
        ta = s2b.function_calls[0].args
        tip_bbox = {"ymin":int(ta["ymin"])+bub_off_y, "xmin":int(ta["xmin"])+bub_off_x, "ymax":int(ta["ymax"])+bub_off_y, "xmax":int(ta["xmax"])+bub_off_x}
    else: tip_bbox = bubble_bbox

    try: zoom_bytes, off_x, off_y, zoom_w, zoom_h = _zoom_region(image_bytes, tip_bbox)
    except Exception: return None, [], "Keynote tip trace failed", []

    # Step 4: Annotate
    print("[Phase 1b] Step 4: Annotate touch point.")
    step4_prompt = (
        f"This is a zoomed-in crop ({zoom_w}×{zoom_h} px) where a '{keyword}' arrow ends.\n\n"
        f"CONFIRMED ARROW LOCATION: The arrow tip was at the centre of this cropped image.\n"
        f"TASK - follow these rules STRICTLY:\n"
        f"  STEP A: Locate the exact spot where the arrowhead TOUCHES the element.\n"
        f"  STEP B: Call `highlight_component` with a tight box around ONLY that element "
        f"at the contact point.\n"
        f"  STEP C: Do NOT annotate anything else - only what the arrow is directly touching.\n\n"
        f"STRICT RULES:\n"
        f"  - Ignore the leader line / arrow shaft entirely.\n"
        f"  - Box the element AT the arrowhead contact point only.\n"
        f"  - Tight fit: thin horizontal line → height = line thickness.\n"
        f"  - Do NOT include surrounding walls, backgrounds, or unrelated elements.\n"
        f"  - Do NOT guess or annotate elements that differ in texture, thickness, or pattern.\n\n"
        f"Coordinates: integers, x∈[0,{zoom_w}], y∈[0,{zoom_h}]."
    )
    s4 = _call_vlm(step4_prompt, zoom_bytes, mime_type="image/jpeg", tools=[highlight_component], tool_config=reveal_cfg, afc_off=True, label="P1b-Step4")

    collected_bboxes = []
    if s4 and s4.function_calls:
        for fc in s4.function_calls:
            if fc.name == "highlight_component":
                a = fc.args
                bbox = {"ymin":int(a["ymin"])+off_y, "xmin":int(a["xmin"])+off_x, "ymax":int(a["ymax"])+off_y, "xmax":int(a["xmax"])+off_x}
                collected_bboxes.append(bbox)
    else:
        collected_bboxes.append({"ymin":max(0,tip_bbox["ymin"]-20), "xmin":max(0,tip_bbox["xmin"]-20), "ymax":min(height,tip_bbox["ymax"]+20), "xmax":min(width,tip_bbox["xmax"]+20)})

    # Step 5: Draw
    print(f"[Phase 1b] Step 5: Drawing annotation → {annotated_path}")
    _draw_boxes_on_image(annotated_path, collected_bboxes, width, height)
    
    # We pass the default description since we removed the text generation step
    desc = f"Keynote {keynote_symbol} pointing to reveal component."
    multi_crops = [{"occurrence_index":1, "annotated_path":annotated_path, "bbox":collected_bboxes[0], "description":desc}]
    return annotated_path, collected_bboxes, desc, multi_crops


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — PDF → Vertex AI
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1(pdf_path: str, search_text: str, page_number: int, instance_index: int = 0, run_id: str = None):
    """Search, highlight, VLM-zoom crop. Returns dict with parts + metadata."""
    update_heartbeat(run_id)
    result = search_text_in_pdf(pdf_path, search_text, page_number, instance_index=instance_index)
    highlighted_pdf = result.get("full_page_path")
    output = {
        "full_pdf_part": None, "cropped_image_part": None, "crop_path": None,
        "full_page_path": highlighted_pdf, "found": result.get("found", False),
        "_search_crop_path": result.get("crop_path"),
    }

    if not (highlighted_pdf and os.path.exists(highlighted_pdf)):
        return output

    with open(highlighted_pdf, "rb") as f:
        output["full_pdf_part"] = types.Part.from_bytes(data=f.read(), mime_type="application/pdf")

    print(f"--- [Phase 1] VLM Zoom for '{search_text}' ---")
    try:
        RASTER_SCALE = 2
        raw_tr = result.get("text_rect");    tr = [v*RASTER_SCALE for v in raw_tr]    if raw_tr else None
        raw_dr = result.get("drawing_rect"); dr = [v*RASTER_SCALE for v in raw_dr] if raw_dr else None
        crop_coords, keynote_symbol = get_vlm_crop(highlighted_pdf, search_text, text_rect=tr, drawing_rect=dr)

        if crop_coords:
            vlm_zoom_path = f"reveal_zoom_{uuid.uuid4().hex[:6]}_{page_number+1}.png"
            if execute_crop(highlighted_pdf, crop_coords, vlm_zoom_path):
                output["crop_path"]      = vlm_zoom_path
                output["crop_coords"]    = crop_coords
                output["keynote_symbol"] = keynote_symbol
                output["keynote_dimension_label"] = get_keynote_dimension(
                    highlighted_pdf, keynote_symbol, search_text) if keynote_symbol else ""
                with open(vlm_zoom_path, "rb") as f:
                    output["cropped_image_part"] = types.Part.from_bytes(data=f.read(), mime_type="image/png")
        else:
            print("--- [Phase 1] VLM no crop — falling back to SearchText crop.")
    except Exception as e:
        print(f"--- [Phase 1] VLM Zoom failed: {e}")

    if not output["cropped_image_part"]:
        crop_path = result.get("crop_path")
        if crop_path and os.path.exists(crop_path):
            output["crop_path"] = crop_path
            with open(crop_path, "rb") as f:
                output["cropped_image_part"] = types.Part.from_bytes(data=f.read(), mime_type="image/png")

    return output


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Drawing Authority Classification
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
    log_response("Phase 2 Classification", raw)
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


def run_phase3(image_part, keyword: str, keynote_symbol: str = "", keynote_dimension_label: str = "", run_id: str = None):
    update_heartbeat(run_id)
    prompt = _EXTRACTION_PROMPT.replace("<<<KEYWORD>>>", keyword)
    prompt += ("\n\nLEGEND DIMENSION (pre-read, confirmed): "
               f"{keynote_dimension_label!r}\nUse as height_value+unit directly."
               if keynote_dimension_label
               else "\n\nLEGEND DIMENSION: not found — use Steps 1-3.")

    resp = _gemini_retry(MODEL_NAME, [prompt, image_part],
                         types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
                         label="Phase 3", run_id=run_id)
    raw = _get_text(resp)
    if not raw: raise RuntimeError("[PHASE 3 ERROR] Gemini returned empty response.")
    log_response("Phase 3 Extraction", raw)
    return _parse_json(raw, required_keys={"height_value","unit","material",
                                           "dimension_label_text","keynote_occurrences"}, phase="PHASE 3")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Verification
# ══════════════════════════════════════════════════════════════════════════════

_VERIFICATION_PROMPT = """You are a Senior Architectural Reviewer verifying extracted <<<KEYWORD>>> dimensions.
Verify the component(s) highlighted with a RED bounding box.

CRITICAL CHECKS:
- Does the extracted height match ONLY the region enclosed by the RED box (not a larger overarching dimension)?
- If multiple RED boxes: extracted height MUST be the correct SUM of all box heights.
- Scale consistency, no contradicting notes (SIM, TYP, REFER).
- Dimension clearly attached to <<<KEYWORD>>>.

Your reason must state whether the dimension matches the top/bottom bounds of the RED box(es).
CRITICAL: Respond ONLY with valid JSON — no explanation, no markdown.
Output: {"verified": <true|false>, "reason": "<justified reason>"}"""


def run_phase4(image_part, extracted_data, keyword: str, run_id: str = None):
    update_heartbeat(run_id)
    sys_prompt = _VERIFICATION_PROMPT.replace("<<<KEYWORD>>>", keyword)
    resp = _gemini_retry(
        MODEL_NAME,
        [f"EXTRACTED_DATA:\n{json.dumps(extracted_data)}", image_part],
        types.GenerateContentConfig(
            system_instruction=sys_prompt, temperature=0.1,
            response_mime_type="application/json",
            response_schema={"type":"object",
                             "properties":{"verified":{"type":"boolean"},"reason":{"type":"string"}},
                             "required":["verified","reason"]},
        ), label="Phase 4", run_id=run_id)
    raw = _get_text(resp)
    if not raw: raise RuntimeError("[PHASE 4 ERROR] Gemini returned empty response.")
    log_response("Phase 4 Verification", raw)
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
    log_response("Phase 6 Method", raw)
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
    log_response("Phase 7 Calculation", raw)
    parsed = _parse_json(raw, phase="PHASE 7")
    parsed = {k.lower(): v for k, v in parsed.items()}
    parsed.setdefault("width", None); parsed.setdefault("height", None); parsed.setdefault("notes", None)
    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Pipeline Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

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
            return {"status":"FAILED","phase2":{"error":"Search text not found"},"phase3":{},"phase4":{},"crop_path":None}

        full_pdf_part       = p1["full_pdf_part"]
        crop_path           = p1.get("crop_path")
        keynote_symbol      = p1.get("keynote_symbol", "")
        keynote_dim_label   = p1.get("keynote_dimension_label", "")
        
        annotated_path      = None
        reveal_bboxes       = []
        reveal_description  = None
        multi_crops         = []

        if crop_path:
            res1b = run_phase1b(crop_path, search_text, keynote_symbol, run_id=run_id)
            annotated_path, reveal_bboxes, reveal_description, multi_crops = res1b
            if not annotated_path:
                print("--- [Phase 1b] Failed — using original crop for final output.")
                annotated_path = crop_path
        else:
            vlm_input_part = full_pdf_part
            multi_crops = [{"occurrence_index": 1, "annotated_path": None, "bbox": {}, "description": "Full image fallback"}]

        # If no multi_crops were generated (e.g. failure in 1b), create a fallback
        if not multi_crops:
            multi_crops = [{"occurrence_index": 1, "annotated_path": annotated_path or crop_path, "bbox": {}, "description": reveal_description or "Fallback"}]

        occurrence_results = []
        for occ in multi_crops:
            idx = occ["occurrence_index"]
            occ_path = occ["annotated_path"]
            print(f"--- [Pipeline] Processing Occurrence {idx} ---")
            
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
            print(f"--- [Phase 2+3] Classifying & Extracting Occ {idx} (parallel) ---")
            def _p2():
                try:    return run_phase2(vlm_input_part, search_text, run_id=run_id)
                except Exception as e: print(_tb.format_exc()); return {"error": f"{type(e).__name__}: {e}"}
            def _p3():
                try:    return run_phase3(vlm_input_part, search_text, keynote_symbol, keynote_dim_label, run_id=run_id)
                except Exception as e: print(_tb.format_exc()); return {"error": f"{type(e).__name__}: {e}"}
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

        # Return the first occurrence as the primary result for backward compatibility,
        # but include the full 'occurrence_results' list.
        primary = occurrence_results[0]
        return {
            "status": "SUCCESS",
            "keynote_symbol": keynote_symbol,
            "keynote_dimension_label": keynote_dim_label,
            "phase1b":  {"bboxes": reveal_bboxes, "description": reveal_description, "multi_crops": multi_crops},
            "occurrence_results": occurrence_results,
            # Backward compatibility fields (from first occurrence)
            "phase2":   primary["phase2"],
            "phase3":   primary["phase3"],
            "crop_path": crop_path,
            "crop_coords": p1.get("crop_coords"),
            "annotated_path": annotated_path,
            "full_page_path": p1.get("full_page_path"),
        }

    except Exception as e:
        print(f"[CRITICAL ERROR] Reveal pipeline failed: {e}")
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
    """
    Scans the PDF for 'reveal' using 3 worker threads.
    Each thread grabs the next page from a shared queue, scans it,
    and processes it if keyword is found — otherwise moves to the next page instantly.
    """
    import queue, threading

    search_keywords = ["Reveal", "REVEAL"]
    _WORKERS = 3

    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()

    print(f"--- [Reveal Pipeline] Starting: {num_pages} pages, {_WORKERS} worker threads.")

    # Shared page queue: every page index goes in, threads pull one at a time
    page_queue = queue.Queue()
    for i in range(num_pages):
        page_queue.put(i)

    # Thread-safe collectors
    results_lock = threading.Lock()
    results = []
    annotated_images = []
    all_temp_files = []

    def _worker(thread_id):
        while True:
            # Check if pipeline has been cancelled
            if is_cancelled():
                print(f"--- [Reveal T{thread_id}] Cancelled — stopping.")
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
                    continue  # no keyword → grab next page immediately

                print(f"--- [Reveal T{thread_id}] Page {page_idx+1} — keyword found, processing...")

                # Process all instances on this page
                seen_on_page = set()
                seen_crops = []
                doc_local = fitz.open(pdf_path)
                page = doc_local[page_idx]
                found_occurrence_on_page = False

                for keyword in search_keywords:
                    if found_occurrence_on_page: break
                    instances = page.search_for(keyword)
                    for inst_idx in range(len(instances)):
                        if found_occurrence_on_page or is_cancelled():
                            break
                        if inst_idx in seen_on_page:
                            continue
                        seen_on_page.add(inst_idx)
                        update_heartbeat(run_id)

                        try:
                            print(f"--- [Reveal T{thread_id}] Page {page_idx+1} inst {inst_idx} '{keyword}' — Phase 1 ---")
                            p1 = run_phase1(pdf_path, keyword, page_idx, instance_index=inst_idx, run_id=run_id)

                            t_files = []
                            if p1.get("full_page_path"): t_files.append(p1["full_page_path"])
                            if p1.get("crop_path"):      t_files.append(p1["crop_path"])
                            if p1.get("_search_crop_path"): t_files.append(p1["_search_crop_path"])

                            if not p1.get("found") or not p1.get("full_pdf_part"):
                                print(f"--- [Reveal T{thread_id}] Page {page_idx+1} inst {inst_idx} — not found, skipping.")
                                with results_lock:
                                    all_temp_files.extend(t_files)
                                continue

                            crop_coords = p1.get("crop_coords")
                            if crop_coords and any(_rects_overlap(crop_coords, prev) for prev in seen_crops):
                                print(f"--- [Reveal T{thread_id}] Page {page_idx+1} inst {inst_idx} — dedup, skipping.")
                                with results_lock:
                                    all_temp_files.extend(t_files)
                                continue
                            if crop_coords:
                                seen_crops.append(crop_coords)

                            print(f"--- [Reveal T{thread_id}] Processing page {page_idx+1} instance {inst_idx} ---")
                            res = run_pipeline(pdf_path, keyword, page_idx, instance_index=inst_idx, run_id=run_id, p1_result=p1)

                            # Re-add to t_files so `all_temp_files` knows about them (but we removed the `os.remove` for these later so they won't be deleted)
                            # Actually, we don't need to add them to `t_files` if we don't want them deleted.
                            # But wait, why wasn't it appending?
                            # Ah, `res` has `annotated_path`. `m_crops` has `annotated_path`.
                            # `a_imgs.append(path)` checks if `os.path.exists(path)`.
                            # IF they are NOT in the debug PDF, maybe `os.path.exists(path)` is False?
                            # Yes! Because `Reveal_Gemini.run_pipeline` might not be saving them correctly, or something else.
                            # Let's check `annotated_path`.
                            
                            m_crops = res.get("phase1b", {}).get("multi_crops", [])
                            a_imgs = []
                            if m_crops:
                                for mc in m_crops:
                                    path = mc.get("annotated_path")
                                    if path and os.path.exists(path):
                                        a_imgs.append(path)
                            else:
                                main_ann = res.get("annotated_path")
                                if main_ann and os.path.exists(main_ann):
                                    a_imgs.append(main_ann)

                            with results_lock:
                                results.append({"page": page_idx, "keyword": keyword, "result": res})
                                annotated_images.extend(a_imgs)
                                all_temp_files.extend(t_files)
                                if res.get("status") == "SUCCESS":
                                    found_occurrence_on_page = True
                                    print(f"--- [Reveal T{thread_id}] Found FIRST occurrence on page {page_idx+1}. Skipping remaining instances on this page.")
                                    break # jump out of the instance loop

                        except Exception as e:
                            import traceback
                            print(f"--- [Reveal T{thread_id}] Page {page_idx+1} inst {inst_idx} failed: {e}")
                            with results_lock:
                                results.append({"page": page_idx, "keyword": keyword, "error": str(e), "traceback": traceback.format_exc()})

                doc_local.close()
                page_queue.task_done()

            except Exception as e:
                import traceback
                print(f"--- [Reveal T{thread_id}] Page {page_idx+1} error: {e}")
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

    print(f"--- [Reveal Pipeline] Done. {len(annotated_images)} annotated images from {len(results)} results.")

    # Build annotated_entries: track which page each annotated image came from
    # so they can be embedded into the debug PDF with sub-page labels (10a, 10b…)
    annotated_entries = []
    page_sub_counter = {}  # page_idx -> next sub-index letter offset
    
    for r in results:
        page_idx = r["page"]
        res = r.get("result", {})
        
        # Collect paths directly from the result object to guarantee ordering
        paths = []
        m_crops = res.get("phase1b", {}).get("multi_crops", [])
        if m_crops:
            for mc in m_crops:
                p = mc.get("annotated_path")
                if p and os.path.exists(p): paths.append(p)
        else:
            p = res.get("annotated_path")
            if p and os.path.exists(p): paths.append(p)
            
        for img_path in paths:
            sub_offset = page_sub_counter.get(page_idx, 0)
            page_sub_counter[page_idx] = sub_offset + 1
            annotated_entries.append({
                "page_idx": page_idx,
                "img_path": img_path,
                "source": "Reveal",
                "sub_idx": sub_offset,
            })

    # Clean up all temporary files after building entries
    for f_path in set(all_temp_files):
        try:
            if os.path.exists(f_path):
                os.remove(f_path)
        except Exception:
            pass

    # Enhanced Logging for outputs
    print(f"\n--- [Reveal Summary] Run ID: {run_id} ---")
    for r in results:
        res = r.get("result", {})
        if res.get("status") == "SUCCESS":
            kn = res.get("keynote_symbol") or "N/A"
            dm = res.get("keynote_dimension_label") or "N/A"
            ds = res.get("phase1b", {}).get("description") or "N/A"
            print(f"Page {r['page']+1} | Keynote: {kn} | Dim: {dm} | Desc: {ds[:100]}...")

    return {
        "status": "SUCCESS" if annotated_images else "NO_MATCHES",
        "annotated_entries": annotated_entries,
        "page_results": results
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) >= 4:
        result = run_pipeline(sys.argv[1], sys.argv[2], int(sys.argv[3]))
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python Reveal_Gemini.py <pdf_path> <search_text> <page_number>")
