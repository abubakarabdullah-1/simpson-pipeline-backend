import json
import os
import re
import ast
import sys
from datetime import datetime
from contextlib import contextmanager


class CaptureOutput(object):
    """Simple output capture that works"""
    def __init__(self):
        self.data = []
    
    def write(self, s):
        self.data.append(s)
    
    def flush(self):
        pass
    
    def getvalue(self):
        return ''.join(self.data)


# Patterns to filter OUT from string log entries (internal/noise/errors)
_NOISE_PATTERNS = re.compile("|".join([
    r"\[Roboflow\]",
    r"\[SAM3\]",
    r"\[VLM\]",
    r"\[VLM Zoom\]",
    r"\[VLM Keynote Refine\]",
    r"\[FALLBACK\]",
    r"Connection error",
    r"Connection refused",
    r"Max retries exceeded",
    r"Failed to connect",
    r"Failed to establish",
    r"Ollama",
    r"VLM Failed",
    r"uploaded to S3",
    r"Upload",
    r"Saved full page to",
    r"Saved.*\.pdf",
    r"Total drawings on page",
    r"Found '.*' \(instance",
    r"Drawing near text",
    r"--- \[Phase",
    r"--- \[Fascia",
    r"--- \[Reveal",
    r"\[Phase 1b?\]",
    r"Phase 1",
    r"pre-computed result",
    r"Rect\(",
    r"PDF coords",
    r"PDF.*pt.*JPEG.*px",
    r"Full page:.*\u2192",
    r"scale=",
    r"keynote='",
    r"zoom_",
    r"fascia_zoom",
    r"reveal_zoom",
    r"reveal_extracted",
    r"\[get_keynote_dimension\]",
    r"keyword found, processing",
    r"Refined keynote crop",
    r"inst \d+ '",
    r"api_key=",
    r"localhost:\d+",
    r"http://localhost",
    r"HTTPConnectionPool",
    r"NewConnectionError",
    r"No masks found",
    r"falling back",
    r"How many leader lines",
    r"\*\*Arrow",
    r"arrow tip",
    r"Pinpointing arrow",
    r"Identifying where",
    r"Identifying Fascia",
    r"Step \d+[a-z]?:",
    r"Step \d+ done",
    r"Bubble Locate",
    r"Tip Trace",
    r"Processing page \d+ instance",
    r"using pre-computed",
    r"--- \[Phase \d\] Complete",
    r"^Processing Page \d+",
    r"^Processing P\d+",
    r"^Calibrating Page \d+",
    r"leader lines",
    r"arrow endpoint",
    r"There are \*\*\d+\*\*",
    r"originating",
    r"^> Processing",
    r"^> Calibrating",
    r"Info: Based on",
    r"Crop saved",
    r"keynote '.*' \u2192",
    r"\[P1b",
    r"Rate limit",
    r"retry in \d+s",
    r"Keynote Flow",
]), re.IGNORECASE)


class LogCollector(object):
    """Captures print statements from phase executions and stores them as structured logs."""
    
    def __init__(self):
        self.logs = {
            "phase1": [],
            "phase2": [],
            "phase3": [],
            "phase4": [],
            "phase5": [],
            "general": []
        }
        self.current_phase = "general"
        self._original_stdout = None
        self._captured_output = None
    
    @contextmanager
    def capture_phase(self, phase_name):
        """Context manager to capture logs for a specific phase."""
        self.current_phase = phase_name
        self._captured_output = CaptureOutput()
        self._original_stdout = sys.stdout
        sys.stdout = self._captured_output
        
        try:
            yield
        finally:
            sys.stdout = self._original_stdout
            captured = self._captured_output.getvalue()
            
            # Parse captured output into log entries
            if captured:
                for line in captured.strip().split('\n'):
                    if line.strip():
                        self.logs[phase_name].append({
                            "timestamp": datetime.utcnow().isoformat(),
                            "message": line.strip()
                        })
            
            # Also print to console
            if captured:
                sys.stdout.write(captured)
    
    def add_log(self, phase_name, message):
        """Manually add a log entry."""
        self.logs[phase_name].append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": message
        })

    @staticmethod
    def _is_noise_string(msg):
        """Check if a string log entry is internal noise."""
        return bool(_NOISE_PATTERNS.search(msg))

    @staticmethod
    def _is_noise_object(obj):
        """Check if a dict log entry is noise/error."""
        # Keep Fascia/Reveal result objects (they have keyword + view)
        if "keyword" in obj and "view" in obj:
            return False
        # Keep schedule/summary objects
        if "action" in obj or "summary" in obj:
            return False
        # Remove FAILED status entries
        if obj.get("status") == "FAILED":
            return True
        # Remove VLM Failed entries
        if obj.get("reason") == "VLM Failed":
            return True
        # Remove phase1-style category classification entries
        if "category" in obj and "reason" in obj and "log" in obj:
            return True
        # Check non-content string values for noise
        for k, v in obj.items():
            if k in ("description", "keyword", "view", "material", "dimension"):
                continue
            if isinstance(v, str) and _NOISE_PATTERNS.search(v):
                return True
        return False

    def _clean_log_list(self, entries):
        """Filter a list of log entries, keeping only clean user-facing items."""
        cleaned = []
        for entry in entries:
            if isinstance(entry, str):
                if not self._is_noise_string(entry):
                    clean = entry.strip()
                    if clean.startswith("> "):
                        clean = clean[2:]
                    if clean.startswith("--- "):
                        continue
                    if clean:
                        cleaned.append(clean)
            elif isinstance(entry, dict):
                if not self._is_noise_object(entry):
                    # Remove internal "log" field
                    clean_obj = {k: v for k, v in entry.items() if k != "log"}
                    cleaned.append(clean_obj)
        return cleaned

    def reorganize_by_page(self):
        """Parses all logs and regroups them by page number."""
        pages = {}
        current_page = "general"
        
        # Regex patterns to detect page numbers
        # Matches: "P1:", "P 1:", "Page 1:", "Processing Page 1"
        page_patterns = [
            r"P\s*(\d+)[:\s]", 
            r"Page\s*(\d+)[:\s]",
            r"Processing Page\s*(\d+)",
            r"Processing P\s*(\d+)",
            r"Calibrating Page\s*(\d+)"
        ]
        
        # Process logs in phase order — skip phase1 (internal classification only)
        phase_order = ["phase2", "phase3", "phase4", "phase5"]
        
        for phase in phase_order:
            if phase not in self.logs: continue
            
            # Reset page context for each phase to avoid bleeding logs from previous phase's last page
            current_page = "general"
            
            for entry in self.logs[phase]:
                msg = entry["message"]
                
                # Check if this line changes the page context
                found_page = None
                for pat in page_patterns:
                    match = re.search(pat, msg, re.IGNORECASE)
                    if match:
                        found_page = match.group(1)
                        current_page = str(found_page)
                        break
                
                # Initialize page entry if needed
                if current_page not in pages:
                    pages[current_page] = {
                        "phase1": [], "phase2": [], "phase3": [], 
                        "phase4": [], "phase5": [], "general": []
                    }
                
                # Structured Parsing per Phase
                
                # Phase 2: Reading (Mode)... or Cataloged X, Y
                if phase == "phase2":
                    p2_match = re.search(r"Reading (.*?)\s*\((.*?)\)", msg)
                    if p2_match:
                        pages[current_page][phase].append({
                            "action": "Reading",
                            "target": p2_match.group(1),
                            "mode": p2_match.group(2),
                            "log": msg
                        })
                    else:
                        cat_match = re.search(r"Cataloged (\d+) Windows, (\d+) Doors", msg)
                        if cat_match:
                            pages[current_page][phase].append({
                                "summary": True,
                                "windows": int(cat_match.group(1)),
                                "doors": int(cat_match.group(2)),
                                "log": msg
                            })
                        else:
                            pages[current_page][phase].append(msg)
                
                # Phase 3: Found {counts}
                elif phase == "phase3":
                    p3_match = re.search(r"- (.*?):\s*Found\s*({.*})", msg)
                    if p3_match:
                        try:
                            counts = ast.literal_eval(p3_match.group(2))
                            pages[current_page][phase].append({
                                "view": p3_match.group(1),
                                "counts": counts,
                                "total_detected": sum(counts.values()),
                                "detected_tags": list(counts.keys()),
                                "log": msg
                            })
                        except:
                            pages[current_page][phase].append(msg)
                    else:
                        pages[current_page][phase].append(msg)
                
                # Phase 4: Scale = ... or FAILED
                elif phase == "phase4":
                    p4_success = re.search(r"- (.*?):\s*SUCCESS.\s*Scale\s*=\s*(.*?)\s*pts/ft", msg)
                    if p4_success:
                        pages[current_page][phase].append({
                            "view": p4_success.group(1),
                            "status": "SUCCESS",
                            "scale": p4_success.group(2),
                            "log": msg
                        })
                    else:
                        p4_fail = re.search(r"- (.*?):\s*FAILED", msg)
                        if p4_fail:
                            pages[current_page][phase].append({
                                "view": p4_fail.group(1),
                                "status": "FAILED",
                                "log": msg
                            })
                        else:
                            pages[current_page][phase].append(msg)

                # Phase 5: Gross ... - Ded ... = Net ...
                elif phase == "phase5":
                    p5_match = re.search(r"- (.*?):\s*Gross\s*(.*?)\s*-\s*Ded\s*(.*?)\s*=\s*Net\s*(.*)", msg)
                    if p5_match:
                        pages[current_page][phase].append({
                            "view": p5_match.group(1),
                            "gross_sf": p5_match.group(2),
                            "deduction_sf": p5_match.group(3),
                            "net_sf": p5_match.group(4),
                            "log": msg
                        })
                    else:
                        pages[current_page][phase].append(msg)

                else:
                    pages[current_page][phase].append(msg)
        
        # Ensure consistent structure and clean all entries
        for page_num in list(pages.keys()):
            # Remove 'general' bucket from each page
            pages[page_num].pop("general", None)
            # Remove phase1 if somehow present
            pages[page_num].pop("phase1", None)
            # Ensure all expected phases exist
            for phase in phase_order:
                if phase not in pages[page_num]:
                    pages[page_num][phase] = []
            # Ensure Fascia/Reveal keys exist (populated later by app.py)
            for kw in ("Fascia", "Reveal"):
                if kw not in pages[page_num]:
                    pages[page_num][kw] = []
            # Clean all log lists to remove noise/errors
            for key in list(pages[page_num].keys()):
                pages[page_num][key] = self._clean_log_list(pages[page_num][key])
        
        # Remove 'general' page bucket
        pages.pop("general", None)
                
        return pages
    
    def save_to_json(self, output_dir, run_id):
        """Save all collected logs to a JSON file."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        log_filename = "{}_logs.json".format(run_id)
        log_path = os.path.join(output_dir, log_filename)
        
        # Reorganize logs by page
        page_logs = self.reorganize_by_page()
        
        log_data = {
            "run_id": run_id,
            "generated_at": datetime.utcnow().isoformat(),
            "pages": page_logs,
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        return log_path
    
    def get_all_logs(self):
        """Return all collected logs."""
        return self.logs
