import cv2
import fitz # PyMuPDF
import numpy as np
from typing import List

def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    """
    Open a PDF and render each page as a numpy array in RGB format.
    """
    images = []
    doc = fitz.open(pdf_path)
    
    # Calculate scale factor for DPI
    # Default DPI in fitz is 72, so scale = dpi / 72
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        # Convert pixmap to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # Ensure it is RGB (3 channels)
        if pix.n == 1: # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif pix.n == 4: # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
        images.append(img)
        
    doc.close()
    return images

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
    
    return cv2.fastNlMeansDenoising(image, h=10)

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
    deskewed = deskew(gray)
    
    # 3. Denoise
    denoised = denoise(deskewed)
    
    # 4. Binarise
    binary = binarise(denoised)
    
    return binary
