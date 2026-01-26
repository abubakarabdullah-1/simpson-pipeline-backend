import ollama
import json
import re
import io

MODEL_NAME = "qwen3-vl:30b-a3b-instruct"

def validate_step(image, candidate_data, phase_context):
    """
    Universal Validator for EIFS Takeoff.
    image: PIL image or bytes
    candidate_data: The JSON output from the phase being tested
    phase_context: String describing what we are validating (e.g., 'Sheet Classification')
    """
    byte_arr = io.BytesIO()
    if not isinstance(image, bytes):
        image.save(byte_arr, format='PNG')
        img_bytes = byte_arr.getvalue()
    else:
        img_bytes = image

    prompt = f"""
    You are a Senior Estimation Auditor.
    
    [AUDIT CONTEXT]: Validating {phase_context}
    [CANDIDATE DATA]: {json.dumps(candidate_data)}
    
    TASK: Inspect the provided image and the candidate data. Verify if the AI has hallucinated.
    
    CRITERIA:
    1. Visual Evidence: Is there actual visual proof in the image for the candidate data?
    2. Logical Consistency: Are dimensions or classifications realistic for a hotel project?
    3. Hallucination Check: Did the AI mention things not present on this specific sheet?
    
    OUTPUT JSON FORMAT (STRICT):
    {{
        "confidence_score": 0.0 to 1.0,
        "status": "VALIDATED" or "WARNING" or "REJECTED",
        "hallucination_detected": true/false,
        "critique": "Detailed explanation of visual discrepancies."
    }}
    """
    
    try:
        response = ollama.chat(
            model=MODEL_NAME, 
            messages=[{'role': 'user', 'content': prompt, 'images': [img_bytes]}],
            format='json'
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        return {
            "confidence_score": 0.0,
            "status": "ERROR",
            "hallucination_detected": True,
            "critique": f"Validation engine failed: {str(e)}"
        }
