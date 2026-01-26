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
            r"Processing P\s*(\d+)"
        ]
        
        # Process logs in phase order usually: 1 -> 2 -> 3 -> 4 -> 5
        phase_order = ["phase1", "phase2", "phase3", "phase4", "phase5"]
        
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
                
                # Phase 1: [Category] (Reason)
                if phase == "phase1":
                    p1_match = re.search(r"\[(.*?)\]\s*\((.*?)\)", msg)
                    if p1_match:
                        pages[current_page][phase].append({
                            "category": p1_match.group(1),
                            "reason": p1_match.group(2),
                            "log": msg
                        })
                    else:
                        pages[current_page][phase].append(msg)
                
                # Phase 2: Reading (Mode)... or Cataloged X, Y
                elif phase == "phase2":
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
        
        # Ensure consistent structure: every page has all phases
        for page_num in pages:
            for phase in phase_order + ["general"]:
                if phase not in pages[page_num]:
                    pages[page_num][phase] = []
        
        # Remove 'general' page as requested by user
        if "general" in pages:
            del pages["general"]
                
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
            "summary": {
                "phase1_count": len(self.logs["phase1"]),
                "phase2_count": len(self.logs["phase2"]),
                "phase3_count": len(self.logs["phase3"]),
                "phase4_count": len(self.logs["phase4"]),
                "phase5_count": len(self.logs["phase5"]),
                "general_count": len(self.logs["general"]),
                "total_logs": sum(len(v) for v in self.logs.values())
            }
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        return log_path
    
    def get_all_logs(self):
        """Return all collected logs."""
        return self.logs
