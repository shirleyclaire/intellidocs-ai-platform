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
