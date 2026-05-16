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

## Retriever & Memory
- **MMR (Maximal Marginal Relevance)**: We use MMR retrieval (`search_type='mmr'`) instead of simple similarity search. MMR selects chunks that are both relevant to the query and diverse from each other, which reduces redundancy in the context provided to the LLM. This is particularly useful when documents have repetitive boilerplate or multiple similar sections.
- **Topic Switch Detection**: Implemented using `sentence-transformers` (specifically `all-MiniLM-L6-v2`). We compute the cosine similarity between the current and previous query. If the similarity falls below `0.4`, we trigger a memory reset. This ensures the assistant doesn't carry over irrelevant context when the user shifts from one policy (e.g., Refunds) to another (e.g., Claims).
- **Conversational Retrieval Chain**: Integrates memory, retrieval, and a custom prompt to provide cited answers.

## Streamlit UI
- **Session State**: Manages `history`, `memory`, `chain`, and `last_question`. The chain is built only after documents are ingested.
- **Sources Expander**: Each assistant response includes a "Sources" expander that lists the file name and page number for every chunk used in the answer.
- **Topic Switch UX**: When a switch is detected, an `st.info()` banner alerts the user that the conversation context has been refreshed.

### 10-Turn Demo Results
The following simulation results verify the pipeline:
1. **Refund Policy?**: Correctly cited Refund Policy section.
2. **Premium customers?**: Answered correctly (Note: topic switch triggered due to specific wording similarity).
3. **Simpler explanation?**: Rephrased the policy accurately.
4. **Grace period?**: Confirmed none mentioned.
5. **Compare policies?**: Synthesized data from both PDF pages.
6. **Claims process?**: **Topic switch detected!** Context cleared for the new topic.
7. **Submit documents?**: Retrieved specific documentation list from Claims process.
8. **Back to refunds?**: **Topic switch detected!** Re-retrieved refund window information.
9. **Summarise refunds**: Provided a concise summary.
10. **Most relevant section**: Correctly identified the Refund Policy section.

### Topic Switch Detection (Turn 6)
```text
> User: Now tell me about the claims process
[INFO] Topic switched — starting fresh context.
Assistant: To file a claim, you should visit our online portal...
```

