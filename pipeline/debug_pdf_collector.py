import os
import re
from typing import List, Tuple, Optional, Dict
from datetime import datetime

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None


# ============================================================
# DEBUG PDF COLLECTOR — INDIVIDUAL PAGE PDFs
# ============================================================
# - Saves each debug image as a separate single-page PDF
# - Named by phase + original page number:
#     phase1_page_10a.pdf, phase2_page_5a.pdf,
#     fascia_page_22a.pdf, reveal_page_15b.pdf
# - Sub-letter (a, b, c…) tracks multiple outputs per page per phase
# - Overlays validator confidence on every page
# - Labels Fascia/Reveal pages with banner text
# - Never crashes pipeline
# ============================================================


def _draw_confidence_overlay(
    img,
    confidence: Optional[float],
    label: Optional[str] = None,
):
    """Draw confidence safely onto image with smaller text."""

    if confidence is None:
        return

    try:
        draw = ImageDraw.Draw(img)

        pct = round(confidence * 100, 2)
        text = "Confidence: {0}%".format(pct)

        if label:
            text = "{0} | {1}".format(label, text)

        # Use medium font size (14pt) - increased from 10pt
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 14)
            except:
                font = None
        
        # Calculate text size based on font
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(text) * 8  # Adjusted for larger font
            text_height = 16
        
        box_w = text_width + 10
        box_h = text_height + 8

        draw.rectangle(
            [5, 5, 5 + box_w, 5 + box_h],
            fill=(255, 255, 255),
        )

        draw.text(
            (8, 8),
            text,
            fill=(0, 0, 0),
            font=font,
        )

    except Exception:
        pass


def _draw_page_label(img, label: str):
    """Draw a prominent page label banner at the top of an annotated image."""
    try:
        draw = ImageDraw.Draw(img)
        W, _ = img.size

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 22)
            except:
                font = None

        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w = len(label) * 13
            text_h = 26

        # Draw a dark banner across the full top
        banner_h = text_h + 16
        draw.rectangle([0, 0, W, banner_h], fill=(30, 30, 30))
        # Centre the text in the banner
        x = max(8, (W - text_w) // 2)
        draw.text((x, 8), label, fill=(255, 255, 255), font=font)

    except Exception:
        pass


def _save_single_page_pdf(img, filepath: str):
    """Save a single PIL image as a one-page PDF."""
    img.save(filepath, "PDF", resolution=150.0)


# ============================================================


def collect_and_write_debug_pdf(
    phase_debug_sets: List[Optional[List[Tuple[object, str]]]],
    output_dir: str,
    *,
    global_confidence: Optional[float] = None,
    per_image_confidence: Optional[Dict[str, float]] = None,
    run_id: Optional[str] = None,
    annotated_entries: Optional[List[Dict]] = None,
    pdf_path: Optional[str] = None,
    phase_start_index: int = 1,
):
    """
    Saves original PDF pages and each debug image as individual
    single-page PDFs in ``<output_dir>/<run_id>/pdf/``.

    Naming convention::

        page_1.pdf            — Original page 1 from uploaded PDF
        page_2.pdf            — Original page 2 from uploaded PDF
        phase1_page_10a.pdf   — Phase 1 output for page 10, first occurrence
        phase2_page_5b.pdf    — Phase 2 output for page 5, second occurrence
        fascia_page_22a.pdf   — Fascia output for page 22, first occurrence
        reveal_page_15b.pdf   — Reveal output for page 15, second occurrence

    Returns:
        str  -> pdf directory path
        None -> nothing generated
    """

    print("[DEBUG-PDF] Collector started")

    try:
        if Image is None:
            print("[DEBUG-PDF] PIL not available — skipping")
            return None

        if not run_id:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        pdf_dir = os.path.join(output_dir, run_id, "pdf")
        os.makedirs(pdf_dir, exist_ok=True)

        saved_files = []

        # ── Save every original page from the uploaded PDF ─────────────────
        if pdf_path and os.path.exists(pdf_path):
            try:
                import fitz  # PyMuPDF
                src_doc = fitz.open(pdf_path)
                for i in range(len(src_doc)):
                    page_filename = "page_{0}.pdf".format(i + 1)
                    page_filepath = os.path.join(pdf_dir, page_filename)
                    single = fitz.open()
                    single.insert_pdf(src_doc, from_page=i, to_page=i)
                    single.save(page_filepath)
                    single.close()
                    saved_files.append(page_filepath)
                print("[DEBUG-PDF] Saved {0} original pages".format(len(src_doc)))
                src_doc.close()
            except Exception as exc:
                print("[DEBUG-PDF] Could not save original pages: {0}".format(exc))

        # Track sub-occurrences: (phase_name, page_num) -> next sub-index
        phase_page_counter = {}

        # ── Phase debug images (only phases with real output) ──────────────
        for phase_idx, debug_list in enumerate(phase_debug_sets, start=phase_start_index):

            if not debug_list:
                print("[DEBUG-PDF] Phase {0}: no output".format(phase_idx))
                continue

            phase_name = "phase{0}".format(phase_idx)

            for entry_idx, entry in enumerate(debug_list):

                try:
                    if not entry or not isinstance(entry, (list, tuple)):
                        continue

                    img = entry[0]
                    label = entry[1] if len(entry) > 1 else None

                    if img is None or not hasattr(img, "save"):
                        continue

                    # Extract page number from label  (e.g. "P10 Schedule Table" -> 10)
                    page_num = None
                    if label:
                        match = re.match(r'^P(\d+)\s', label)
                        if match:
                            page_num = int(match.group(1))

                    if page_num is None:
                        page_num = entry_idx + 1  # fallback

                    # Clean label — remove the P{N} prefix for overlay text
                    clean_label = label
                    if label:
                        clean_label = re.sub(r'^P\d+\s+', '', label)

                    # Confidence
                    conf = None
                    if per_image_confidence and label:
                        conf = per_image_confidence.get(label)
                    if conf is None:
                        conf = global_confidence

                    # Prepare image copy
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img_copy = img.copy()

                    _draw_confidence_overlay(img_copy, conf, clean_label)

                    # Sub-letter (a, b, c…) for multiple outputs on the same page
                    key = (phase_name, page_num)
                    sub_idx = phase_page_counter.get(key, 0)
                    phase_page_counter[key] = sub_idx + 1
                    sub_letter = chr(ord("a") + sub_idx)

                    # Save as individual PDF
                    filename = "{0}_page_{1}{2}.pdf".format(phase_name, page_num, sub_letter)
                    filepath = os.path.join(pdf_dir, filename)

                    _save_single_page_pdf(img_copy, filepath)
                    saved_files.append(filepath)
                    print("[DEBUG-PDF] Saved: {0}".format(filename))

                except Exception as exc:
                    print(
                        "[DEBUG-PDF] Phase {0} entry {1} skipped: {2}".format(
                            phase_idx, entry_idx, exc
                        )
                    )

        # ── Annotated Fascia / Reveal pages ────────────────────────────────
        if annotated_entries:
            _save_annotated_entries(annotated_entries, pdf_dir, saved_files)

        if not saved_files:
            print("[DEBUG-PDF] No debug images saved")
            return None

        print("[DEBUG-PDF] Saved {0} individual PDFs to {1}".format(len(saved_files), pdf_dir))
        return pdf_dir

    except Exception as exc:
        print("[DEBUG-PDF] Fatal suppressed: {0}".format(exc))
        return None


def _save_annotated_entries(
    annotated_entries: List[Dict],
    pdf_dir: str,
    saved_files: List[str],
):
    """
    Save Fascia/Reveal annotated images as individual PDFs into *pdf_dir*.

    Checks existing files in the directory so sub-letters continue correctly
    when Fascia and Reveal finish at different times.
    """
    sorted_entries = sorted(
        annotated_entries,
        key=lambda e: (e.get("page_idx", 0), e.get("sub_idx", 0), e.get("source", "")),
    )

    # Scan existing files to pick up where we left off with sub-letters
    source_page_counter = {}
    for fname in os.listdir(pdf_dir):
        m = re.match(r'^(fascia|reveal)_page_(\d+)([a-z])\.pdf$', fname)
        if m:
            src = m.group(1)
            pg = int(m.group(2))
            letter_val = ord(m.group(3)) - ord("a") + 1
            key = (src, pg)
            source_page_counter[key] = max(source_page_counter.get(key, 0), letter_val)

    for entry in sorted_entries:
        try:
            img_path = entry.get("img_path", "")
            if not img_path or not os.path.exists(img_path):
                continue

            human_page = entry.get("page_idx", 0) + 1
            source = entry.get("source", "Annotated").lower()

            key = (source, human_page)
            sub_idx = source_page_counter.get(key, 0)
            source_page_counter[key] = sub_idx + 1
            sub_letter = chr(ord("a") + sub_idx)

            with Image.open(img_path) as ann_img:
                if ann_img.mode != "RGB":
                    ann_img = ann_img.convert("RGB")
                ann_copy = ann_img.copy()

            filename = "{0}_page_{1}{2}.pdf".format(source, human_page, sub_letter)
            filepath = os.path.join(pdf_dir, filename)

            _save_single_page_pdf(ann_copy, filepath)
            saved_files.append(filepath)
            print("[DEBUG-PDF] Saved annotated: {0}".format(filename))

        except Exception as exc:
            print("[DEBUG-PDF] Annotated entry skipped: {0}".format(exc))


def append_annotated_to_debug_pdf(
    existing_pdf_path: Optional[str],
    annotated_entries: List[Dict],
    output_dir: str,
    run_id: Optional[str] = None,
) -> Optional[str]:
    """
    Saves Fascia/Reveal annotated images as individual PDFs in the
    ``<output_dir>/<run_id>/pdf/`` directory.

    This replaces the old behaviour of appending to a single combined
    debug PDF.  The function signature is kept for backward-compatible
    call-sites in app.py.

    Returns the pdf directory path, or *existing_pdf_path* on failure.
    """
    if not annotated_entries:
        return existing_pdf_path

    if Image is None:
        print("[DEBUG-PDF] PIL not available — cannot save annotated pages")
        return existing_pdf_path

    try:
        if not run_id:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        pdf_dir = os.path.join(output_dir, run_id, "pdf")
        os.makedirs(pdf_dir, exist_ok=True)

        saved_files = []
        _save_annotated_entries(annotated_entries, pdf_dir, saved_files)

        if saved_files:
            print("[DEBUG-PDF] Saved {0} annotated PDFs to {1}".format(len(saved_files), pdf_dir))

        return pdf_dir

    except Exception as exc:
        print("[DEBUG-PDF] append_annotated_to_debug_pdf failed: {0}".format(exc))
        import traceback; traceback.print_exc()
        return existing_pdf_path
