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

## Ingestion Pipeline
- **Chunking Strategy**: Used `RecursiveCharacterTextSplitter` with `chunk_size=500` and `chunk_overlap=100`. The 500-character size ensures each vector captures a distinct, focused thought (like a single policy rule), while the 100-character overlap prevents sentences from being cut off awkwardly across chunks, maintaining context.
- **Metadata**: Every chunk includes a `source_file` (basename of the original document) and a `chunk_index` (integer ID of the chunk). This is critical for the RAG assistant to cite its sources explicitly.
- **Test Results**: The ingestion pipeline successfully loaded, chunked, and stored documents in a ChromaDB instance, with similarity search returning the correct chunks and preserving the required metadata.
