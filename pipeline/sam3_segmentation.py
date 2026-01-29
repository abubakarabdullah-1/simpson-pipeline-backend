"""
SAM3 Segmentation Module for Building Mask Generation

This module provides a wrapper around the trained SAM3 model for
automatic and prompt-based segmentation of building elevations.
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, List
import io

# ==========================================
# CONFIGURATION
# ==========================================
# ROBOFLOW CONFIGURATION
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "http://localhost:9001").rstrip("/")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "w6CLKytmY6ccsKUC82GG")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "version-2-rk7ss/2")

# ==========================================
# ROBOFLOW INFERENCE CLIENT
# ==========================================
import requests
import base64
import json

def roboflow_infer(image_array, prompts=["building", "elevation"]):
    """
    Send inference request to Roboflow Docker container (optimized for speed).
    """
    try:
        # Downscale if too large (speed optimization)
        h, w = image_array.shape[:2]
        if max(h, w) > 1920:
            scale = 1920 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image_array = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Fast JPEG encoding with lower quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR), encode_param)
        img_str = base64.b64encode(buffer).decode("utf-8")
        base64_img = f"data:image/jpeg;base64,{img_str}"
        
        # Prepare payload - handle list of prompts
        if isinstance(prompts, str):
            prompts = [prompts]
            
        prompt_payload = [{"type": "text", "text": p} for p in prompts]
        
        payload = {
            "image": {"type": "base64", "value": base64_img},
            "model_id": ROBOFLOW_MODEL_ID,
            "prompts": prompt_payload,
            "confidence": 0.10,      # Lower threshold to catch more
            "iou_threshold": 0.10    # Lower IoU threshold
        }
        
        url = f"{ROBOFLOW_API_URL}/sam3/concept_segment?api_key={ROBOFLOW_API_KEY}"
        print(f"[Roboflow] Sending request to {ROBOFLOW_API_URL}...")
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            masks = []
            
            # Parsing logic based on user's working snippet
            # Structure: result -> prompt_results -> predictions -> masks (polygons)
            try:
                prompt_results = result.get('prompt_results', [])
                
                for p_result in prompt_results:
                    # Get the prompt text for this group of predictions
                    current_prompt = "Result"
                    if 'prompt' in p_result and 'text' in p_result['prompt']:
                        current_prompt = p_result['prompt']['text'].capitalize()
                    
                    predictions = p_result.get('predictions', [])
                    
                    for pred in predictions:
                        # 'masks' contains list of polygons (contours)
                        contours = pred.get('masks', [])
                        
                        for contour in contours:
                            pts = np.array(contour, dtype=np.int32)
                            
                            # Create binary mask
                            mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
                            cv2.fillPoly(mask, [pts], 255)
                            
                            area = cv2.contourArea(pts)
                            x, y, w, h = cv2.boundingRect(pts)
                            
                            masks.append({
                                'segmentation': mask > 0,
                                'area': area,
                                'bbox': [x, y, w, h],
                                'predicted_iou': pred.get('confidence', 1.0),
                                'stability_score': pred.get('confidence', 1.0),
                                'class_name': current_prompt
                            })
                            
                print(f"[Roboflow] Success! Detected {len(masks)} objects")
                return masks
                
            except Exception as e:
                print(f"[Roboflow] Error parsing response: {e}")
                print(f"[Roboflow] Raw result: {json.dumps(result)}")
                return None

        else:
            print(f"[Roboflow] Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"[Roboflow] Connection error: {e}")
        return None

# ==========================================
# AUTOMATIC MASK GENERATION
# ==========================================
def segment_building_automatic(image: np.ndarray, prompt="elevation", min_area_ratio=0.005, max_area_ratio=0.95) -> Tuple[Optional[np.ndarray], List[dict]]:
    """
    Segment building using Roboflow Docker API only.
    """
    try:
        masks_info = roboflow_infer(image, prompts=[prompt])
        
        if masks_info and len(masks_info) > 0:
            img_area = image.shape[0] * image.shape[1]
            valid_masks = []
            
            for mask_info in masks_info:
                area = mask_info['area']
                area_ratio = area / img_area
                
                if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                    continue
                valid_masks.append(mask_info)
            
            if not valid_masks:
                return None, []
            
            # Use largest valid mask as primary
            best_mask_info = max(valid_masks, key=lambda x: x['area'])
            best_mask = best_mask_info['segmentation'].astype(np.uint8) * 255
            return best_mask, valid_masks
            
    except Exception as e:
        print(f"[Roboflow] Exception: {e}")
    
    return None, []

# ==========================================
# UNUSED / UNSUPPORTED METHODS
# ==========================================
def segment_building_with_points(image, point_coords, point_labels):
    print("[SAM3] Point segmentation not supported with Roboflow Concept API")
    return None

def segment_building_with_box(image, box):
    print("[SAM3] Box segmentation not supported with Roboflow Concept API")
    return None

# ==========================================
# OPENCV FALLBACK
# ==========================================
def segment_building_opencv_fallback(gray_image: np.ndarray, dilation_px=50) -> Tuple[np.ndarray, tuple]:
    _, binary = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray_image)
    if not contours: return mask, gray_image.shape[::-1]
    
    img_area = gray_image.shape[0] * gray_image.shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > (img_area * 0.005) and area < (img_area * 0.95):
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
    if dilation_px > 0:
        kernel = np.ones((dilation_px, dilation_px), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
    return mask, (gray_image.shape[1], gray_image.shape[0])

# ==========================================
# UNIFIED INTERFACE
# ==========================================
def generate_building_mask(image: np.ndarray, use_sam3=True, dilation_px=50) -> Tuple[np.ndarray, tuple]:
    # Use Roboflow (labeled as use_sam3 to keep API consistent)
    if use_sam3:
        # Convert to RGB if needed for API
        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            gray_image = image
        else:
            rgb_image = image
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        mask, _ = segment_building_automatic(rgb_image)
        if mask is not None:
            if dilation_px > 0:
                kernel = np.ones((dilation_px, dilation_px), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
            return mask, (mask.shape[1], mask.shape[0])
            
    # Fallback to OpenCV
    print("[Pipeline] Using OpenCV fallback")
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return segment_building_opencv_fallback(image, dilation_px)

