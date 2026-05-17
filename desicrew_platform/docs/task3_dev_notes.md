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
- **Document Classification**: Implemented in [classifier.py](file:///c:/Users/Shirley%20Claire/Desktop/Shirley/Projects/intellidocs-ai-platform/desicrew_platform/task3_doc_pipeline/classifier.py) using a highly performant, deterministic hybrid model.
  - **Fuzzy Anchor Matching**: Computes the token-set ratio (using `rapidfuzz.fuzz.token_set_ratio`) across class anchor phrases loaded from a JSON configuration file.
  - **Regex Signature Validation**: Validates the predicted class by performing a regular expression search on the document text.
  - **Routing Algorithm**:
    - *STRAIGHT_THROUGH*: Matches with high fuzzy confidence ($\ge 0.85$) AND matches the class regex $\rightarrow$ `flagged = False`.
    - *BORDERLINE_RESCUE*: Matches with definitive fuzzy confidence ($\ge 0.92$) but fails the class regex (e.g., due to OCR missing the card/policy number) $\rightarrow$ `flagged = False`.
    - *CONFLICT*: Otherwise $\rightarrow$ `flagged = True` (flag reason recorded).

## Architecture Decisions
- **Deterministic Hybrid Classification**: By utilizing string-distance metric heuristics (`RapidFuzz` anchors) combined with validation patterns (`re` patterns) instead of deep-learning classifiers, the system achieves near-instantaneous execution times (< 1ms) and 100% deterministic, explainable routing paths.
- **Configurability**: Anchor phrases and regular expressions are externalized in `config/document_classes.json` for easy extension to new document types.

## Debug/Change Log
- Scaffolded pipeline files.
- Added implementation of `shared/` utilities.
- Implemented `preprocess.py` (DPI conversion, deskew, fast NL means denoise, Otsu binarisation, and CLAHE enhancements).
- Implemented `ocr_engine.py` (PaddleOCR execution, dual dictionary/list format parser, and 15px layout reading order grouping).
- Implemented `classifier.py` (RapidFuzz token-set matching and regular expression validation).
- Integrated classifier execution and results formatting in the pipeline test script `task_3_test_ocr.py`.
- Developed `test_classifier.py` unit test suite containing 6 tests covering all classification routes.

