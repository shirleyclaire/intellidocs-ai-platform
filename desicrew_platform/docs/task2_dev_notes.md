# Task 2: Document-Aware RAG Assistant - Developer Notes

## Implementation Log
- Initialized empty project structure.
- Documented shared dependencies.

## Architecture Decisions
- Segregated into `app.py` (UI), `ingest.py` (document processing), `retriever.py` (RAG logic), and `memory.py` (conversation state).

## Shared Dependencies
- **`shared.llm`**: Used for the response synthesis generation step.
- **`shared.embeddings`**: Instantiates `sentence-transformers` for embedding generation during ingestion and retrieval.
- **`shared.vector_store`**: Handles Chroma connection and `add_documents` logic.
- **`shared.prompts`**: Uses `RAG_SYSTEM_PROMPT` to enforce citation, rule compliance, and source bounds.
- **`shared.utils`**: Uses `get_file_extension` and file loading utilities for ingestion step.
- **Design Choices**: Offloading ChromaDB lifecycle and prompt management to `shared/` reduces repetition and guarantees all pipeline elements use the exact same embedding model.

## Feature Notes
- Pending: PDF/DOCX/TXT ingestion.
- Pending: RAG logic with citation requirements.
- Pending: Conversation memory and topic tracking.

## Debug/Change Log
- Scaffolded files.
- Added implementation of `shared/` utilities.

## Known Limitations
- (TBD)
