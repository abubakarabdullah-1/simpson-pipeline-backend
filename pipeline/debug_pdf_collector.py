import os
import re
from typing import List, Tuple, Optional, Dict
from datetime import datetime

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None


# ============================================================
# DEBUG PDF COLLECTOR — VALIDATOR DRIVEN
# ============================================================
# - Combines debug images from any subset of phases
# - Skips None / empty / broken entries
# - Overlays validator confidence on EVERY page
# - Appends annotated Fascia/Reveal pages with sub-page labels
# - Never crashes pipeline
# - Returns pdf path or None
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


# ============================================================


def collect_and_write_debug_pdf(
    phase_debug_sets: List[Optional[List[Tuple[object, str]]]],
    output_dir: str,
    *,
    global_confidence: Optional[float] = None,
    per_image_confidence: Optional[Dict[str, float]] = None,
    run_id: Optional[str] = None,
    annotated_entries: Optional[List[Dict]] = None,
):
    """
    Returns:
        str  -> pdf path
        None -> nothing generated

    annotated_entries: list of dicts with keys:
        page_idx  (int)  — 0-based page number in the source PDF
        img_path  (str)  — path to the annotated image file
        source    (str)  — "Fascia" or "Reveal"
        sub_idx   (int)  — 0-based occurrence index on that page (0→a, 1→b …)
    """

    print("[DEBUG-PDF] Collector started")

    try:
        if Image is None:
            print("[DEBUG-PDF] PIL not available — skipping")
            return None

        images: List[Image.Image] = []

        for phase_idx, debug_list in enumerate(phase_debug_sets, start=1):

            if not debug_list:
                print("[DEBUG-PDF] Phase {0}: no output".format(phase_idx))
                continue

            for entry_idx, entry in enumerate(debug_list):

                try:
                    if not entry or not isinstance(entry, (list, tuple)):
                        continue

                    img = entry[0]
                    label = entry[1] if len(entry) > 1 else None

                    if img is None or not hasattr(img, "save"):
                        continue
                    
                    # Remove page number prefix (e.g., "P1 ", "P2 ") from label
                    if label:
                        label = re.sub(r'^P\d+\s+', '', label)

                    conf = None

                    if per_image_confidence and label:
                        conf = per_image_confidence.get(label)

                    if conf is None:
                        conf = global_confidence

                    # Preserve original image - create a copy to avoid modifying original
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    # Create copy to preserve original quality
                    img_copy = img.copy()

                    _draw_confidence_overlay(img_copy, conf, label)

                    images.append(img_copy)

                except Exception as exc:
                    print(
                        "[DEBUG-PDF] Phase {0} entry {1} skipped: {2}".format(
                            phase_idx, entry_idx, exc
                        )
                    )

        # ── Annotated Fascia / Reveal pages ────────────────────────────────
        if annotated_entries:
            # Sort: by source page index, then sub_idx, then source name
            sorted_entries = sorted(
                annotated_entries,
                key=lambda e: (e.get("page_idx", 0), e.get("sub_idx", 0), e.get("source", "")),
            )

            for entry in sorted_entries:
                try:
                    img_path = entry.get("img_path", "")
                    if not img_path or not os.path.exists(img_path):
                        continue

                    # page_idx is 0-based; humans read pages starting from 1
                    human_page = entry.get("page_idx", 0) + 1
                    sub_letter = chr(ord("a") + entry.get("sub_idx", 0))
                    source = entry.get("source", "Annotated")
                    banner = f"{source} — Page {human_page}{sub_letter}"

                    with Image.open(img_path) as ann_img:
                        if ann_img.mode != "RGB":
                            ann_img = ann_img.convert("RGB")
                        ann_copy = ann_img.copy()

                    _draw_page_label(ann_copy, banner)
                    images.append(ann_copy)
                    print(f"[DEBUG-PDF] Appended annotated page: {banner}")

                except Exception as exc:
                    print(f"[DEBUG-PDF] Annotated entry skipped: {exc}")

        if not images:
            print("[DEBUG-PDF] No debug images collected")
            return None

        os.makedirs(output_dir, exist_ok=True)

        if not run_id:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        pdf_path = os.path.join(output_dir, "{0}_debug.pdf".format(run_id))

        first, rest = images[0], images[1:]

        first.save(
            pdf_path,
            save_all=True,
            append_images=rest,
            resolution=150.0,  # Preserve quality
        )

        print(
            "[DEBUG-PDF] PDF created: {0} ({1} pages)".format(
                pdf_path, len(images)
            )
        )

        return pdf_path

    except Exception as exc:
        print("[DEBUG-PDF] Fatal suppressed: {0}".format(exc))
        return None


def append_annotated_to_debug_pdf(
    existing_pdf_path: Optional[str],
    annotated_entries: List[Dict],
    output_dir: str,
    run_id: Optional[str] = None,
) -> Optional[str]:
    """
    Appends labelled Fascia/Reveal annotated images to an existing debug PDF.
    Uses fitz (PyMuPDF) to merge so the existing pages are preserved.

    annotated_entries: list of dicts with keys:
        page_idx  (int)  — 0-based page number in the source PDF
        img_path  (str)  — path to the annotated image file
        source    (str)  — "Fascia" or "Reveal"
        sub_idx   (int)  — 0-based occurrence index on that page (0→a, 1→b …)

    Returns the (possibly updated) pdf path, or existing_pdf_path unchanged on failure.
    """
    if not annotated_entries:
        return existing_pdf_path

    if Image is None:
        print("[DEBUG-PDF] PIL not available — cannot append annotated pages")
        return existing_pdf_path

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[DEBUG-PDF] fitz not available — cannot merge PDFs")
        return existing_pdf_path

    try:
        # ── Build new annotated pages (PIL images with banner label) ──────────
        sorted_entries = sorted(
            annotated_entries,
            key=lambda e: (e.get("page_idx", 0), e.get("sub_idx", 0), e.get("source", "")),
        )

        new_images: List[Image.Image] = []
        # Keep track of which new image belongs to which original page index
        img_to_page_idx = []
        for entry in sorted_entries:
            try:
                img_path = entry.get("img_path", "")
                if not img_path or not os.path.exists(img_path):
                    continue

                human_page = entry.get("page_idx", 0) + 1
                sub_letter = chr(ord("a") + entry.get("sub_idx", 0))
                source = entry.get("source", "Annotated")
                banner = f"{source} — Page {human_page}{sub_letter}"

                with Image.open(img_path) as ann_img:
                    if ann_img.mode != "RGB":
                        ann_img = ann_img.convert("RGB")
                    ann_copy = ann_img.copy()

                _draw_page_label(ann_copy, banner)
                new_images.append(ann_copy)
                img_to_page_idx.append(entry.get("page_idx", 0))
                print(f"[DEBUG-PDF] Appended annotated page: {banner}")

            except Exception as exc:
                print(f"[DEBUG-PDF] Annotated entry skipped: {exc}")

        if not new_images:
            return existing_pdf_path

        # ── Save new annotated images to a temporary PIL PDF ─────────────────
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_ann_pdf = tmp.name

        first_ann, rest_ann = new_images[0], new_images[1:]
        first_ann.save(
            tmp_ann_pdf,
            save_all=True,
            append_images=rest_ann,
            resolution=150.0,
        )

        # ── Merge: existing debug PDF + new annotated PDF via fitz ───────────
        os.makedirs(output_dir, exist_ok=True)
        if not run_id:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        pdf_path = os.path.join(output_dir, f"{run_id}_debug.pdf")

        merged = fitz.open()

        # How many original pages are there? 
        # The base PDF starts with Phase 1 overview pages (one for each original page).
        # We assume the first N pages in existing_pdf_path correspond to the N original pages.
        # This is true because collect_and_write_debug_pdf writes Phase 1 first.
        
        # We need a robust insertion logic.
        # Phase 1 overview page for page N is always the (N+1)th page in the ORIGINAL unmodified debug PDF (0-indexed).
        # However, if Fascia finishes first, it inserts pages. If Reveal finishes later, it inserts pages.
        # So we can't just assume Phase 1 for page idx N is at index N.
        # We must track how many extra pages we've already inserted.
        # Let's search for "Phase 1 - Page " text in the PDF to find the anchor pages.
        # Wait, the PDF pages are rasterized images. `get_text()` returns nothing for them.
        
        # Since reading text is impossible, we can use the page dimensions!
        # NO. We can just use an offset map. But app.py doesn't track it.
        # Actually, let's keep it simple: we insert the new pages *right after* the (page_idx)th page 
        # of the CURRENT document, assuming that the first N pages of the debug PDF 
        # roughly correspond to the N pages. We will calculate an insertion index.
        # Since `app.py` usually batches Fascia+Reveal if they finish before Main,
        # they are both appended at the same time. If they finish after Main, they are appended sequentially.

        if existing_pdf_path and os.path.exists(existing_pdf_path):
            with fitz.open(existing_pdf_path) as existing_doc:
                merged.insert_pdf(existing_doc)
            print(f"[DEBUG-PDF] Loaded {len(merged)} existing page(s) from {existing_pdf_path}")
            
            # Group new images by their target anchor page index
            from collections import defaultdict
            pages_by_idx = defaultdict(list)
            
            with fitz.open(tmp_ann_pdf) as ann_doc:
                for idx, target_page_idx in enumerate(img_to_page_idx):
                    pages_by_idx[target_page_idx].append(idx)
                    
                # We iterate backwards through the pages_by_idx so insertions don't shift earlier targets
                for target_page_idx in sorted(pages_by_idx.keys(), reverse=True):
                    # In a newly minted debug PDF, the anchor page for `target_page_idx` is exactly `target_page_idx`.
                    # We want to insert the annotated pages immediately after it, i.e., at `insert_idx = target_page_idx + 1`.
                    # But what if there are already some annotated pages there from a previous run?
                    # E.g., Fascia was inserted, and now Reveal is doing it.
                    # We can use a simple heuristic: just insert at `target_page_idx + 1`. 
                    # If other pages were inserted there before, it pushes them down, but they stay grouped near the original page!
                    insert_idx = min(target_page_idx + 1, len(merged))
                    
                    for ann_doc_page_num in pages_by_idx[target_page_idx]:
                        merged.insert_pdf(ann_doc, from_page=ann_doc_page_num, to_page=ann_doc_page_num, start_at=insert_idx)
                        insert_idx += 1 # Advance so next appended page goes after the one we just inserted

        else:
            # No existing PDF, just save the annotated ones
            with fitz.open(tmp_ann_pdf) as ann_doc:
                merged.insert_pdf(ann_doc)

        merged.save(pdf_path)
        merged.close()

        # Clean up temp file
        try:
            os.remove(tmp_ann_pdf)
        except Exception:
            pass

        total = fitz.open(pdf_path)
        n_pages = len(total)
        total.close()
        print(f"[DEBUG-PDF] Updated debug PDF: {pdf_path} ({n_pages} pages total)")
        return pdf_path

    except Exception as exc:
        print(f"[DEBUG-PDF] append_annotated_to_debug_pdf failed: {exc}")
        import traceback; traceback.print_exc()
        return existing_pdf_path

