# Task 3: Document Extraction Pipeline - Developer Notes

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
