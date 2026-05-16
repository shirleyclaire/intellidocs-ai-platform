# Task 3: Document Extraction Pipeline - Developer Notes

## Pre-processing Pipeline
The accuracy of OCR (Optical Recognition) depends heavily on the quality of the input image. Scanned insurance forms often suffer from skew, noise, and uneven lighting. We implement a multi-stage pre-processing pipeline to normalize images before passing them to the OCR engines.

### 1. PDF to Image Conversion
- **Tool**: PyMuPDF (`fitz`).
- **Logic**: Renders PDF pages as high-resolution (300 DPI) pixmaps and converts them to numpy arrays. High DPI is critical for capturing small text in dense forms.

### 2. Deskewing
- **Logic**: Calculates the skew angle of the text by finding the minimum area rectangle containing all non-white pixels.
- **Threshold**: Only applies rotation if the detected angle is > 0.5 degrees to avoid rotating correctly aligned documents due to sub-pixel noise.
- **Benefit**: Ensures that text lines are perfectly horizontal, which is required for efficient layout analysis and line-by-line OCR.

### 3. Denoising
- **Logic**: Uses `cv2.fastNlMeansDenoising`.
- **Benefit**: Removes "salt and pepper" noise and scanning artifacts that could be mistaken for punctuation or small characters.

### 4. Binarisation
- **Logic**: Adaptive Thresholding (`cv2.adaptiveThreshold` with Gaussian window).
- **Why Adaptive vs. Otsu's?**:
    - **Otsu's Method**: Calculates a global threshold. This fails when a document has uneven lighting or variable ink density (common in handwritten forms or scans with shadows).
    - **Adaptive Thresholding**: Calculates the threshold for every pixel based on its local neighborhood. This handles gradients and shadows much more robustly, preserving text clarity across the entire page.

### 5. Final Output
The pipeline produces a single-channel binary image (values 0 or 255 only) which maximizes contrast for the OCR engine.

## Implementation Log
- Initialized empty project structure.
- Documented shared dependencies.

## Architecture Decisions
- Pipeline structure: classification, OCR (Tesseract + PaddleOCR), extraction, validation, scoring, output.

## Shared Dependencies
- **`shared.ocr`**: Wraps both PyTesseract and PaddleOCR engines to run against images. Standardizes OCR confidence format and error boundaries.
- **`shared.prompts`**: Uses `CLASSIFIER_PROMPT` for the document classification phase.
- **`shared.llm`**: Employed for structural extraction phase, converting OCR text into structured JSON.
- **`shared.utils`**: Used for writing the final results with `save_json`.
- **Design Choices**: Consolidating OCR dependencies in `shared.ocr` isolates `pytesseract` and `paddleocr` failures, presenting a unified interface for the extraction pipeline.

## Feature Notes
- Pending: Document classification.
- Pending: OCR processing based on text type (printed vs handwritten).
- Pending: Extraction logic with confidence scoring.
- Pending: Human review flagging mechanism.

## Debug/Change Log
- Scaffolded files.
- Added implementation of `shared/` utilities.

## Known Limitations
- (TBD)
