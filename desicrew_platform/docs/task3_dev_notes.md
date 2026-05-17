# Task 3: Document Extraction Pipeline - Developer Notes

## Pre-processing Pipeline
The accuracy of OCR (Optical Recognition) depends heavily on the quality of the input image. Scanned insurance forms often suffer from skew, noise, and uneven lighting. We implement a multi-stage pre-processing pipeline to normalize images before passing them to the OCR engines.

### 1. PDF to Image Conversion
- **Tool**: `pdf2image` (using Poppler).
- **Logic**: Converts PDF pages into high-fidelity 300 DPI PIL Images. High DPI is critical for capturing small details and text on compact documents like ID cards.

### 2. Deskewing
- **Tool**: `deskew` library.
- **Logic**: Calculates the skew angle of the text using `deskew.determine_skew()`.
- **Threshold**: Only applies affine rotation (via OpenCV) if the absolute detected angle is >= 0.5 degrees to avoid unnecessary rotation of correctly aligned documents.
- **Benefit**: Standardizes document alignment to horizontal, crucial for both Tesseract and PaddleOCR line segmentations.

### 3. Denoising
- **Tool**: OpenCV `cv2.fastNlMeansDenoising` with smoothing parameter `h=10`.
- **Benefit**: Removes scanner grain, salt-and-pepper noise, and fine artifacts that could result in OCR character recognition errors.

### 4. Binarisation
- **Tool**: Otsu's Thresholding (`cv2.threshold` with `cv2.THRESH_BINARY + cv2.THRESH_OTSU`).
- **Benefit**: Automatically determines the optimal threshold value by minimizing intra-class variance of the black and white pixels, maximizing contrast for reliable character extraction.

### 5. Final Output
- **Format**: 3-channel RGB PIL Image (reconverted from binary single-channel).
- **Reason**: PaddleOCR requires 3-channel (RGB) input format, so the binarised 1-channel image is cast back to RGB space to maintain full compatibility across engines.

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
