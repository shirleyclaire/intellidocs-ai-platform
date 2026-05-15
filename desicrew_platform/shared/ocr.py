"""OCR Utilities using Tesseract and PaddleOCR."""

import numpy as np
import pytesseract
from typing import Dict, Any, Optional

# Lazy-loaded PaddleOCR instance
_PADDLE_INSTANCE = None

def _get_paddle():
    """Helper to lazy-load PaddleOCR."""
    global _PADDLE_INSTANCE
    if _PADDLE_INSTANCE is None:
        from paddleocr import PaddleOCR
        _PADDLE_INSTANCE = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    return _PADDLE_INSTANCE

def run_tesseract(image: np.ndarray, psm: int = 6) -> Dict[str, Any]:
    """
    Run PyTesseract OCR on an image.
    
    Args:
        image (np.ndarray): The image array.
        psm (int): Page segmentation mode.
        
    Returns:
        dict: A dictionary containing extracted text, confidence, and engine name.
    """
    try:
        data = pytesseract.image_to_data(
            image, 
            config=f'--psm {psm}', 
            output_type=pytesseract.Output.DICT
        )
        
        valid_confs = []
        words = []
        
        # 'conf' list contains strings or ints. We process them safely.
        for i, conf in enumerate(data.get('conf', [])):
            try:
                c = float(conf)
                if c != -1:
                    valid_confs.append(c)
                    words.append(str(data['text'][i]))
            except (ValueError, TypeError):
                continue
                
        text = " ".join(words).strip()
        confidence = sum(valid_confs) / len(valid_confs) / 100.0 if valid_confs else 0.0
        
        return {
            "text": text,
            "confidence": confidence,
            "engine": "tesseract"
        }
    except Exception:
        return {
            "text": "",
            "confidence": 0.0,
            "engine": "tesseract"
        }

def run_paddle(image: np.ndarray) -> Dict[str, Any]:
    """
    Run PaddleOCR on an image.
    
    Args:
        image (np.ndarray): The image array.
        
    Returns:
        dict: A dictionary containing extracted text, confidence, and engine name.
    """
    try:
        ocr = _get_paddle()
        result = ocr.ocr(image, cls=True)
        
        lines = []
        confs = []
        
        if result and result[0]:
            for line in result[0]:
                if len(line) == 2 and isinstance(line[1], (list, tuple)) and len(line[1]) == 2:
                    text, conf = line[1]
                    lines.append(str(text))
                    confs.append(float(conf))
                    
        text = "\n".join(lines).strip()
        confidence = sum(confs) / len(confs) if confs else 0.0
        
        return {
            "text": text,
            "confidence": confidence,
            "engine": "paddle"
        }
    except Exception:
        return {
            "text": "",
            "confidence": 0.0,
            "engine": "paddle"
        }
