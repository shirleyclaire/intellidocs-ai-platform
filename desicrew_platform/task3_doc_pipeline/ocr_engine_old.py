import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import List, Dict, Any
from shared.ocr import run_tesseract, run_paddle

# def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
#     """
#     Open a PDF and render each page as a numpy array in RGB format.
#     """
#     images = []

    
#     # Calculate scale factor for DPI
#     # Default DPI in fitz is 72, so scale = dpi / 72
#     zoom = dpi / 72
#     mat = fitz.Matrix(zoom, zoom)
    
#     for page in doc:
#         pix = page.get_pixmap(matrix=mat)
#         # Convert pixmap to numpy array
#         img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
#         # Ensure it is RGB (3 channels)
#         if pix.n == 1: # Grayscale
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         elif pix.n == 4: # RGBA
#             img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
#         images.append(img)
        
#     doc.close()
#     return images

def deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct image skew by rotating to align text horizontally.
    Expects grayscale image.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    # Threshold to find non-white pixels
    # We want text to be foreground (white on black for minAreaRect)
    # So we invert the image: pixels < 200 become foreground
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find coordinates of non-zero pixels
    coords = np.column_stack(np.where(binary > 0))
    
    if len(coords) == 0:
        return image
        
    # Compute the minimum area rectangle angle
    angle = cv2.minAreaRect(coords)[-1]
    
    # minAreaRect returns angle in range [-90, 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Only rotate if significant skew
    if abs(angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    
    return image

def denoise(image: np.ndarray) -> np.ndarray:
    """
    Apply fast non-local means denoising. Input must be grayscale.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    return cv2.fastNlMeansDenoising(image, h=3) # changed h=10 to h=3

def binarise(image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding for robust binarization. Input must be grayscale.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    return cv2.adaptiveThreshold(
        image, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 2
    )

def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Main pre-processing pipeline: grayscale -> deskew -> denoise -> binarise.
    """
    # 1. Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    # 2. Deskew
    # deskewed = deskew(gray)
    deskewed = gray 
    
    # 3. Denoise
    denoised = denoise(deskewed)
    
    # 4. Binarise
    binary = binarise(denoised)
    
    return binary

def get_tesseract_confidence(image: np.ndarray) -> float:
    """
    Calculate the mean confidence of Tesseract OCR on the given image.
    Returns 0.0 - 1.0.
    """
    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(image)
    
    try:
        # Get detailed OCR data
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        
        # Extract confidence values, filtering out -1 (no text detected)
        confidences = [float(c) for c in data['conf'] if float(c) != -1]
        
        if not confidences:
            return 0.0
            
        return sum(confidences) / len(confidences) / 100.0
    except Exception:
        return 0.0

def is_handwritten_region(image: np.ndarray, conf_threshold: float = 0.40) -> bool:
    """
    Determine if a region is likely handwritten based on Tesseract confidence.
    """
    conf = get_tesseract_confidence(image)
    return conf < conf_threshold

def ocr_region(image: np.ndarray, force_handwritten: bool = False) -> Dict[str, Any]:
    """
    Perform OCR on a region, routing to the appropriate engine.
    """
    # Decide which engine to use initially
    if force_handwritten or is_handwritten_region(image):
        result = run_paddle(image)
    else:
        result = run_tesseract(image, psm=6)
        
    # If confidence is too low, try the other engine
    if result["confidence"] < 0.3:
        alt_engine = "paddle" if result["engine"] == "tesseract" else "tesseract"
        
        if alt_engine == "paddle":
            alt_result = run_paddle(image)
        else:
            alt_result = run_tesseract(image, psm=6)
            
        # Return the result with higher confidence
        if alt_result["confidence"] > result["confidence"]:
            return alt_result
            
    return result

def ocr_full_document(image: np.ndarray) -> Dict[str, Any]:
    """
    Perform OCR on the full document with intelligent routing.
    Prioritizes Tesseract for speed, falling back to Paddle only if needed.
    """

    # To run paddleocr alone uncomment the below line
    result = run_paddle(image)


    # # 1. Try Tesseract first (Fast Path)
    # # PSM 6 (Single block of text) is usually good for structured documents
    # result = run_tesseract(image, psm=6)

    # # 2. Fallback to Paddle only if Tesseract confidence is low (< 40%) or empty
    # if result["confidence"] < 0.4 or not result["text"].strip():
    #     print("Note: Low Tesseract confidence, attempting PaddleOCR fallback...")
    #     paddle_result = run_paddle(image)
        
    #     # If Paddle actually got something better, use it
    #     if paddle_result["confidence"] > result["confidence"]:
    #         return {
    #             "full_text": paddle_result["text"],
    #             "confidence": paddle_result["confidence"],
    #             "engine": "paddle"
    #         }
        
    return {
        "full_text": result["text"],
        "confidence": result["confidence"],
        "engine": result["engine"]
    }
