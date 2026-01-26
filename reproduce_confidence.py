import sys
import os

# Add local directory to path so we can import pipeline
sys.path.append(os.getcwd())

from unittest.mock import MagicMock
sys.modules['fitz'] = MagicMock()
sys.modules['ollama'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['PIL.ImageDraw'] = MagicMock()
sys.modules['numpy'] = MagicMock()

from pipeline.runner import compute_confidence

def test_confidence():
    print("--- Testing Confidence Score Calculation ---")

    # Scenario 1: Perfect Run (100%)
    # 2 pages, both surveyed, both scaled, windows & doors found, grand total > 0
    s1_elevations = [1, 2]
    s1_specs = {"windows": {"W1": {}}, "doors": {"D1": {}}}
    s1_survey = {
        1: {"View 1": {"W1": 5}}, 
        2: {"View 2": {"D1": 2}}
    }
    s1_scale = {
        1: {"View 1": 50.0},
        2: {"View 2": 50.0}
    }
    s1_items = []
    s1_total = 1000
    
    score1 = compute_confidence(s1_elevations, s1_specs, s1_survey, s1_scale, s1_items, s1_total)
    print(f"Scenario 1 (Perfect): Expected 100.0, Got {score1}")
    assert score1 == 100.0

    # Scenario 2: Partial Run
    # 2 pages. 
    # Specs: Only Windows (10/20 pts). 
    # Survey: 1/2 pages (12.5/25 pts). 
    # Scale: 1/2 pages (12.5/25 pts).
    # Elev Found: Yes (20 pts).
    # Grand Total: Yes (10 pts).
    # Expected: 20 + 10 + 12.5 + 12.5 + 10 = 65.0
    s2_elevations = [1, 2]
    s2_specs = {"windows": {"W1": {}}} # No doors
    s2_survey = {
        1: {"View 1": {"W1": 5}}, 
        2: {} # Empty
    }
    s2_scale = {
        1: {"View 1": 50.0},
        2: {} # Empty
    }
    s2_items = []
    s2_total = 500
    
    score2 = compute_confidence(s2_elevations, s2_specs, s2_survey, s2_scale, s2_items, s2_total)
    print(f"Scenario 2 (Partial): Expected 65.0, Got {score2}")
    assert score2 == 65.0

    # Scenario 3: Failed Run
    # No elevations.
    # Expected: 0
    s3_elevations = []
    s3_specs = {}
    s3_survey = {}
    s3_scale = {}
    s3_items = []
    s3_total = 0
    
    score3 = compute_confidence(s3_elevations, s3_specs, s3_survey, s3_scale, s3_items, s3_total)
    print(f"Scenario 3 (Fail): Expected 0, Got {score3}")
    assert score3 == 0

    print("--- ALL TESTS PASSED ---")

if __name__ == "__main__":
    test_confidence()
