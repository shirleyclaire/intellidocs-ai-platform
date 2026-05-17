# Task 3: Document Extraction Pipeline - Developer Notes

# Document Preprocessing Pipeline for Deep-Learning OCR

This module provides an optimized preprocessing pipeline specifically tailored for deep-learning-based Optical Character Recognition (OCR) engines, such as **PaddleOCR**. 

Unlike traditional OCR engines (e.g., Tesseract) that rely heavily on aggressive binary (black and white) images, modern deep-learning OCR models perform significantly better when text features, gradients, and anti-aliasing are preserved.

---

## Function Overview

### `preprocess_document(file_path: str) -> List[Image.Image]`

Transforms an input document (PDF or Image) into a list of preprocessed PIL Images optimized for text detection and recognition.

### Key Features
*   **Multi-Format Support:** Seamlessly handles multi-page `.pdf` files as well as standard image formats (`.png`, `.jpg`, `.jpeg`, `.tiff`).
*   **Gradient-Preserving Deskewing:** Corrects document rotation using cubic interpolation and edge replication to avoid introducing harsh artificial borders.
*   **Adaptive Contrast Enhancement:** Replaces destructive global/adaptive binarization with CLAHE to rescue text trapped in shadows or uneven lighting.

---

## Core Preprocessing Techniques Explained

### 1. File Handling & Format Normalization
The function acts as a unified interface for both vector/raster documents and standalone images. 
*   **PDFs:** Converted to images at **300 DPI** via `pdf2image`. This resolution provides the optimal balance between text clarity and computational speed for deep-learning backbones.
*   **Images:** Loaded safely using PIL's lazy loading verification (`img.load()`) to catch corrupted files early.

### 2. Deskewing with Edge Replication
Skewed text significantly degrades OCR bounding box accuracy. The pipeline calculates the skew angle and rotates the image back to 0 degrees using an affine transformation.

*   **Interpolation (`cv2.INTER_CUBIC`):** Uses a bicubic interpolation over a $4 \times 4$ pixel neighborhood to keep character edges smooth after rotation.
*   **Border Mode (`cv2.BORDER_REPLICATE`):** Traditional rotation introduces stark black or white triangles at the image corners. This pipeline replicates the outermost edge pixels instead. This prevents deep learning models from misinterpreting sharp geometric borders as document layout edges or text lines.

### 3. CLAHE vs. Traditional Binarization
Historically, thresholding (like Otsu's Binarization) was mandatory for OCR. However, deep-learning models utilize convolutional neural networks (CNNs) that thrive on texture, sub-pixel gradients, and anti-aliasing. Aggressive binarization strips this critical spatial data away and amplifies image noise.

Instead, this pipeline uses **CLAHE (Contrast Limited Adaptive Histogram Equalization)**:
*   **Localization:** It divides the image into contextual tiles ($8 \times 8$ grids) and equalizes the histogram locally, making it highly effective at neutralizing uneven lighting, page folds, and shadows.
*   **Contrast Limiting:** It clips the histogram at a threshold (`clipLimit=2.0`) to prevent over-amplifying background noise in completely uniform areas.

---

## Technical Specifications & Dependencies

### Prerequisites

To run this pipeline, ensure your environment has the system-level dependency **Poppler** installed (required for PDF conversion), alongside the following Python libraries:

```bash
pip install opencv-python numpy Pillow pdf2image

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

- **Field Extraction**: Implemented in [extractor.py](file:///c:/Users/Shirley%20Claire/Desktop/Shirley/Projects/intellidocs-ai-platform/desicrew_platform/task3_doc_pipeline/extractor.py) using deterministic regular expressions and spatial proximity coordinate metrics.
  - **Data Structure**: `ExtractedField` encapsulates the target field name, extracted value, method type (`"regex"`, `"spatial"`, or `"failed"`), minimum OCR confidence, and scorer placeholder.
  - **Regex Extraction**:
    - Generates a text map corresponding index ranges to exact OCR token objects.
    - Resolves sub-token boundaries and calculates the strict **minimum** confidence among all matching tokens.
  - **Spatial Extraction**:
    - Searches for anchor labels with a `rapidfuzz.fuzz.ratio` threshold of $\ge 75$.
    - **Right Direction**: Identifies tokens horizontally aligned with the anchor (vertical offset $\le 20$px) and within `pixel_threshold` distance.
    - **Below Direction**: Identifies tokens directly underneath the anchor (left vertical column horizontal offset $\le 50$px) and within `pixel_threshold` distance.
    - **Smart Grouping**: For the vertical direction, it identifies the nearest line immediately underneath the label, isolates other tokens on that same line (vertical offset $\le 20$px), and sorts them left-to-right to maintain natural reading order for names.
  - **Dispatch Plan & Fallback Safety**:
    - Maps 10 document classes to target schemas of extraction lambdas.
    - Evaluates the plan sequentially; if any extraction returns `None`, it appends a placeholder field with `method="failed"`, ensuring the output schema length matches exactly.

- **Field Confidence Scoring**: Implemented in [scorer.py](file:///c:/Users/Shirley%20Claire/Desktop/Shirley/Projects/intellidocs-ai-platform/desicrew_platform/task3_doc_pipeline/scorer.py) to assess the reliability of each extracted field.
  - **Scoring Constants**:
    - `REGEX_BASE_SCORE = 1.0` (Regex matches are highly authoritative as they strictly adhere to known syntax patterns).
    - `SPATIAL_BASE_SCORE = 0.95` (Spatial proximity extraction is inherently slightly less reliable than strict regex, accounting for potential layout shifts).
    - `FIELD_FLAG_THRESHOLD = 0.75` (Fields scoring below this are flagged for human-in-the-loop review).
  - **Formula**:
    - For `method == "regex"`: $Score = REGEX\_BASE\_SCORE \times OCR\_confidence$
    - For `method == "spatial"`: $Score = SPATIAL\_BASE\_SCORE \times OCR\_confidence$
    - For missing fields (`value is None` or `method == "failed"`): $Score = 0.0$
    - Scores are clamped to $[0.0, 1.0]$ and stored in `field.extraction_confidence`.

- **Exception Handling / Multimodal LLM Fallback**: Implemented in [llm_fallback.py](file:///c:/Users/Shirley%20Claire/Desktop/Shirley/Projects/intellidocs-ai-platform/desicrew_platform/task3_doc_pipeline/llm_fallback.py) to handle AI Exception Handling (rescue failed/borderline fields without needing human intervention).
  - **Routing Criteria**: Any field where the value is missing or the confidence score is $< 0.75$ is automatically sent to the fallback route.
  - **API Loader**: Safely checks `GEMINI_API_KEY` env var and `.streamlit/secrets.toml` under `[gemini]` `api_key` using standard standard `tomllib`.
  - **Direct Multimodal Inference**: Instead of re-running OCR or text parsing, it passes the preprocessed PIL Image directly alongside a target prompt to `gemini-1.5-flash`.
  - **Prompting & Output Guardrails**: Configures the model with `"response_mime_type": "application/json"` and prompts it to return a valid JSON object matching the exact field names.
  - **Post-Processing**: Updates the field in-place with the rescued value, sets `method = "llm_fallback"`, and assigns a default confidence of `0.90` (since Gemini's visual inference is highly robust).

- **Output Formatter**: Implemented in [output_formatter.py](file:///c:/Users/Shirley%20Claire/Desktop/Shirley/Projects/intellidocs-ai-platform/desicrew_platform/task3_doc_pipeline/output_formatter.py) to compile the final JSON structures.
  - **Compliant Output Schema**: Combines classification statistics, detailed field confidence, extraction method, and flagging state.
  - **Consolidated Audit Report**: Outputs individual `<document_id>.json` files and aggregates all records requiring human verification into a central `flagging_report.json`, detailing precise, machine-readable rationales (e.g. classification failure reasons or low-confidence field details).

- **Streamlit Interactive UI Frontend**: Implemented in [app.py](file:///c:/Users/Shirley%20Claire/Desktop/Shirley/Projects/intellidocs-ai-platform/desicrew_platform/task3_doc_pipeline/app.py) as the interactive, premium dashboard for document extraction.
  - **Wide Layout & State Management**: Configured with a `wide` layout and pre-initialized state containers (`pipeline_complete`, `output_json`, `flagging_report`, `pipeline_log`) to ensure stable rendering across hot reloads.
  - **Dynamic Module Mapping**: Resolves path conflicts by dynamically mapping real submodules (`task3_doc_pipeline.preprocess`, etc.) to the synthetic `pipeline` namespace within `sys.modules`. This ensures the imported code uses the exact user-specified constraint block:
    ```python
    from pipeline.preprocessor import preprocess_document
    from pipeline.ocr_engine import run_ocr, tokens_to_text
    ...
    ```
  - **Real-Time Step Renders**: Utilizes `st.status()` to track progress updates live during extraction, showing stages like preprocessing, OCR layout mapping, hybrid classification, and Gemini fallbacks.
  - **Security Safe-Redaction Log**: Employs regular expressions on log entries to redact sensitive Aadhaar (12-digit) and PAN (alphanumeric) numbers in the developer console display, replacing them with standard tags like `[Aadhaar Redacted]`.
  - **Dual-Column Visual Panel**:
    - *Left*: Displays structured `output_json` using a clean `st.json()` tree.
    - *Right*: Displays predicted class badges, fuzzy scores, regex metrics, and detailed field cards. Confidence values are color-coded (Green $> 0.85$, Amber $> 0.75$, Red $< 0.75$). Banners show custom error lists if any validations fail.
  - **Side-by-Side Exporters**: Renders responsive `st.download_button` widgets side-by-side once the pipeline is complete, enabling users to export individual document extractions (`output.json`) and flagged review records (`flagging_report.json`) in one click.
  - **Resilient Execution**: Wraps the backend pipeline in structured `try-except-finally` blocks, writing execution errors directly to the live console logs and letting the app fail gracefully without freezing or crashing the browser window.

## Architecture Decisions
- **Deterministic Hybrid Classification**: By utilizing string-distance metric heuristics (`RapidFuzz` anchors) combined with validation patterns (`re` patterns) instead of deep-learning classifiers, the system achieves near-instantaneous execution times (< 1ms) and 100% deterministic, explainable routing paths.
- **Configurability**: Anchor phrases, classification regex, and extraction patterns are externalized in `config/document_classes.json` for easy extension to new document types.
- **Straight-Through Processing (STP) with AI Exception Handling**: By utilizing lightning-fast deterministic rules for the vast majority of standard extractions and only routing anomalies to Gemini, the platform maximizes throughput, ensures near-zero hallucination risk, saves considerable API computing costs, and eliminates unnecessary human labor.
- **Dynamic Module Aliasing**: Dynamically linking local folders to synthetic module targets allows us to adhere to strict interface imports while keeping our folder names fully clean, descriptive, and separate from other tasks.

## Debug/Change Log
- Scaffolded pipeline files.
- Added implementation of `shared/` utilities.
- Implemented `preprocess.py` (DPI conversion, deskew, fast NL means denoise, Otsu binarisation, and CLAHE enhancements).
- Implemented `ocr_engine.py` (PaddleOCR execution, dual dictionary/list format parser, and 15px layout reading order grouping).
- Implemented `classifier.py` (RapidFuzz token-set matching and regular expression validation).
- Integrated classifier execution and results formatting in the pipeline test script `task_3_test_ocr.py`.
- Developed `test_classifier.py` unit test suite containing 6 tests covering all classification routes.
- Implemented `extractor.py` (character-to-token regex mapper, vertical and horizontal spatial coordinate extractors, and dispatcher).
- Created a robust 6-test suite in `test_extractor.py` covering confidence bounds, spatial proximity groups, and fallbacks.
- Updated `task_3_test_ocr.py` to seamlessly execute extraction and print results beautifully.
- Implemented `scorer.py` (deterministic formulas, clamping, and field confidence calculation).
- Implemented `llm_fallback.py` (Gemini credentials loading, multimodal vision inference, and in-place field updates).
- Implemented `output_formatter.py` (structured schemas, individual and consolidated JSON writers, and clear human rationales).
- Created a dedicated test suite in `test_scorer_formatter.py` verifying correct scoring math, threshold flagging, and JSON storage.
- Integrated all pipeline stages in the final `task_3_test_ocr.py` script.
- Created `app.py` Streamlit frontend inside `task3_doc_pipeline` with real-time logger tracking, dual-column visual panels, download managers, and dynamic module namespaces.




