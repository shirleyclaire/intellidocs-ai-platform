# Desicrew AI Platform

## Project Overview
A modular AI platform with three independent applications designed for data analysis, document-aware question answering (RAG), and advanced document extraction.

## Tech Stack
| Component | Technology |
|---|---|
| UI Framework | Streamlit |
| Data Processing | Pandas |
| LLM Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace (sentence-transformers) |
| OCR | Tesseract, PaddleOCR |

## Setup Instructions
1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Configure `.streamlit/secrets.toml` with `OPENROUTER_API_KEY`

## Architecture Notes
The platform is separated into three distinct tasks. A `shared` folder contains utilities that serve all three applications.

## What was implemented in `shared/`
- **LLM Utilities:** Centralized OpenRouter/ChatOpenAI instantiation with local Ollama fallback logic.
- **Utilities:** Common file and string manipulation functions (`load_file`, `save_json`, `clean_text`, etc.).
- **Prompts:** Standardized prompt strings for all three applications.
- **Embeddings:** Cached instantiation of `HuggingFaceEmbeddings`.
- **Vector Store:** ChromaDB connection logic and document persistence mechanisms.
- **OCR Engine:** Wrappers for `pytesseract` and `PaddleOCR` with error handling and confidence normalization.

## Task 1 — Excel Agent
This is a Streamlit application designed to act as an automated data analyst.
- **How to run**: `streamlit run task1_excel_agent/app.py`
- **What it does**: 
  - Allows you to upload `.xlsx` and `.xls` files.
  - Automatically loads the data into a Pandas dataframe.
  - Exposes a chat interface where you can ask analytical questions in plain English.
  - Writes and executes underlying Pandas code (which you can preview!) to deliver the answers.

## Task 2 — Document Support Assistant
A RAG-based assistant for querying policy documents.
- **How to run**: `streamlit run desicrew_platform/task2_rag_assistant/app.py`
- **What it does**:
  - Ingests PDF, DOCX, and TXT files.
  - Uses MMR retrieval for diverse context.
  - Features semantic topic switching to clear context when moving between unrelated questions.
  - Provides full source citations (file name and page number).

## Task 3 — Document Extraction Pipeline
An advanced document processing and OCR extraction pipeline for handling diverse physical identity cards and documents.
- **How to test preprocessing, OCR, & Classification**: Run the combined pipeline visualization test with:
  ```bash
  venv/Scripts/python desicrew_platform/task_3_test_ocr.py
  ```
- **How to run unit tests**:
  - Run the OCR engine tests:
    ```bash
    venv/Scripts/python -c "import sys; sys.path.append('desicrew_platform'); import unittest; unittest.main(module='test_ocr_engine')"
    ```
  - Run the document hybrid classifier tests:
    ```bash
    venv/Scripts/python -c "import sys; sys.path.append('desicrew_platform'); import unittest; unittest.main(module='test_classifier')"
    ```
- **What it does**:
  - **Pre-processing Stage**:
    - Validates document formats: PDF, PNG, JPG, JPEG, TIFF/TIF.
    - Converts PDF pages into high-fidelity 300 DPI images (using `pdf2image` and `poppler`).
    - Applies a sequential cleaning pipeline: grayscale conversion, automated deskewing (via `deskew.determine_skew()`), fast non-local means denoising (`h=10`), local contrast enhancement (CLAHE), and conversion back to 3-channel RGB space.
  - **OCR Engine Stage** ([ocr_engine.py](file:///c:/Users/Shirley%20Claire/Desktop/Shirley/Projects/intellidocs-ai-platform/desicrew_platform/task3_doc_pipeline/ocr_engine.py)):
    - Uses PaddleOCR as the sole extraction engine (no Tesseract fallback).
    - Supports both the legacy list-of-lists format and the new PaddleX dictionary key parallel lists format.
    - Tokenizes text into `OCRToken` objects containing `text`, absolute `bbox` coordinate tuple `(x_min, y_min, x_max, y_max)`, `confidence` score, and `page` index.
    - Filters tokens using a module-level threshold `OCR_CONFIDENCE_FLOOR = 0.40`.
    - Concatenates tokens into readable text using `tokens_to_text()`: sorts tokens top-to-bottom and left-to-right by `y_min` (using a 15-pixel tolerance line group) and then by `x_min`.
  - **Classification Stage** ([classifier.py](file:///c:/Users/Shirley%20Claire/Desktop/Shirley/Projects/intellidocs-ai-platform/desicrew_platform/task3_doc_pipeline/classifier.py)):
    - Determines which of 10 document classes the document belongs to by loading config rules from [document_classes.json](file:///c:/Users/Shirley%20Claire/Desktop/Shirley/Projects/intellidocs-ai-platform/desicrew_platform/config/document_classes.json).
    - Utilizes a deterministic hybrid of fuzzy anchor scoring (via `rapidfuzz` token-set ratio) and regular expression pattern matching.
    - Implements STRAIGHT-THROUGH matching and BORDERLINE-RESCUE routing logic to flag low-confidence or non-compliant documents automatically.
  - **Extraction Stage** ([extractor.py](file:///c:/Users/Shirley%20Claire/Desktop/Shirley/Projects/intellidocs-ai-platform/desicrew_platform/task3_doc_pipeline/extractor.py)):
    - Extracts target fields dynamically using a robust dispatch plan mapped to the 10 document classes.
    - Performs **Regex Extraction** with index character-to-token mappings to capture full matching bounds and preserve minimum OCR confidence.
    - Performs **Spatial Extraction** with fuzzy anchor resolution ($\ge 75$ match ratio) and proximity constraints horizontally (`right`) or vertically (`below`).
    - Uses smart horizontal line isolation for vertical collection, grouping adjacent terms together in a single row (tolerance: 20px) and sorting left-to-right.
    - Employs a critical fallback guarantee to return placeholders (`method="failed"`) for any missing fields to preserve exact output schema lengths.

- **How to run unit tests**:
  - Run the OCR engine tests:
    ```bash
    venv/Scripts/python -c "import sys; sys.path.append('desicrew_platform'); import unittest; unittest.main(module='test_ocr_engine')"
    ```
  - Run the document hybrid classifier tests:
    ```bash
    venv/Scripts/python -c "import sys; sys.path.append('desicrew_platform'); import unittest; unittest.main(module='test_classifier')"
    ```
  - Run the field extractor tests:
    ```bash
    venv/Scripts/python -c "import sys; sys.path.append('desicrew_platform'); import unittest; unittest.main(module='test_extractor')"
    ```




