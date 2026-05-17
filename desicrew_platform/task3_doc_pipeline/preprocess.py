import os
import cv2
import numpy as np
from PIL import Image
from typing import List
from deskew import determine_skew

# def preprocess_document(file_path: str) -> List[Image.Image]:
#     """
#     Accepts a file path (PDF or image). Returns a list of preprocessed PIL Image objects, one per page.

#     Steps applied:
#     1. Convert PDF to images at 300 DPI, or load image directly.
#     2. Convert to grayscale.
#     3. Deskew using deskew library.
#     4. Denoise using fastNlMeansDenoising (h=10).
#     5. Binarise using Otsu's thresholding.
#     6. Convert back to RGB PIL Image.
#     """
#     filename = os.path.basename(file_path)
#     _, ext = os.path.splitext(file_path)
#     ext = ext.lower()

#     # Supported extensions check
#     valid_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif'}
#     if ext not in valid_extensions:
#         raise ValueError(
#             f"Unsupported file format: File '{filename}' has invalid extension '{ext}'. "
#             f"Supported extensions are: {sorted(list(valid_extensions))}"
#         )

#     # 1. Load document pages as PIL Images
#     pages: List[Image.Image] = []
#     if ext == '.pdf':
#         try:
#             from pdf2image import convert_from_path
#             pages = convert_from_path(file_path, dpi=300)
#         except Exception as e:
#             raise RuntimeError("pdf2image failed: ensure poppler is installed and on PATH.") from e
#     else:
#         try:
#             img = Image.open(file_path)
#             # Force loading the image data to catch any file read/corruption issues early
#             img.load()
#             pages = [img]
#         except Exception as e:
#             raise IOError(f"Failed to load image file '{filename}': {e}")

#     preprocessed_pages: List[Image.Image] = []

#     # 2. Process each page image
#     for page in pages:
#         # Convert to grayscale
#         # Using PIL to convert to RGB first guarantees that 1-channel or palette images
#         # are properly converted before getting converted to grayscale numpy array.
#         img_rgb = np.array(page.convert('RGB'))
#         img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

#         # Deskewing
#         # Apply deskew.determine_skew() and rotate to correct angle.
#         # If skew angle is None or < 0.5 degrees, skip rotation.
#         angle = determine_skew(img_gray)
#         if angle is not None and abs(angle) >= 0.5:
#             (h, w) = img_gray.shape[:2]
#             center = (w // 2, h // 2)
#             M = cv2.getRotationMatrix2D(center, angle, 1.0)
#             # Use BORDER_CONSTANT with 255 (white) to fill empty rotated borders cleanly
#             img_gray = cv2.warpAffine(
#                 img_gray, M, (w, h),
#                 flags=cv2.INTER_CUBIC,
#                 borderMode=cv2.BORDER_CONSTANT,
#                 borderValue=255
#             )

#         # Denoising
#         # Apply cv2.fastNlMeansDenoising (h=10).
#         denoised = cv2.fastNlMeansDenoising(img_gray, h=10)

#         # Binarisation
#         # Apply Otsu threshold via cv2.threshold with cv2.THRESH_BINARY + cv2.THRESH_OTSU.
#         _, binarised = cv2.threshold(
#             denoised, 0, 255,
#             cv2.THRESH_BINARY + cv2.THRESH_OTSU
#         )

#         # Convert back to RGB PIL Image (PaddleOCR requires 3-channel input)
#         binarised_rgb = cv2.cvtColor(binarised, cv2.COLOR_GRAY2RGB)
#         processed_pil = Image.fromarray(binarised_rgb)
#         preprocessed_pages.append(processed_pil)

#     return preprocessed_pages


def preprocess_document(file_path: str) -> List[Image.Image]:
    """
    Optimized preprocessing for deep-learning OCR (PaddleOCR).
    Replaces aggressive binarisation with CLAHE contrast enhancement.
    """
    filename = os.path.basename(file_path)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    valid_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    if ext not in valid_extensions:
        raise ValueError(
            f"Unsupported file format: File '{filename}' has invalid extension '{ext}'. "
            f"Supported extensions are: {sorted(list(valid_extensions))}"
        )

    pages: List[Image.Image] = []
    if ext == '.pdf':
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(file_path, dpi=300)
        except Exception as e:
            raise RuntimeError("pdf2image failed: ensure poppler is installed and on PATH.") from e
    else:
        try:
            img = Image.open(file_path)
            img.load()
            pages = [img]
        except Exception as e:
            raise IOError(f"Failed to load image file '{filename}': {e}")

    preprocessed_pages: List[Image.Image] = []

    for page in pages:
        img_rgb = np.array(page.convert('RGB'))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Deskewing
        angle = determine_skew(img_gray)
        if angle is not None and abs(angle) >= 0.5:
            (h, w) = img_gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_gray = cv2.warpAffine(
                img_gray, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE # Better than constant white for deep learning
            )

        # 2. Local Contrast Enhancement (CLAHE) instead of Binarisation
        # This brings out text hidden in shadows without destroying gradients
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(img_gray)

        # 3. Convert back to RGB for PaddleOCR
        final_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        processed_pil = Image.fromarray(final_rgb)
        preprocessed_pages.append(processed_pil)

    return preprocessed_pages