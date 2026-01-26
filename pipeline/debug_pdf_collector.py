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


# ============================================================


def collect_and_write_debug_pdf(
    phase_debug_sets: List[Optional[List[Tuple[object, str]]]],
    output_dir: str,
    *,
    global_confidence: Optional[float] = None,
    per_image_confidence: Optional[Dict[str, float]] = None,
    run_id: Optional[str] = None,
):
    """
    Returns:
        str  -> pdf path
        None -> nothing generated
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
