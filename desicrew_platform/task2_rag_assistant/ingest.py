"""Ingestion pipeline for Document-Aware RAG Assistant."""
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from shared.vector_store import get_or_create_store, add_documents

def load_document(path: str) -> List[Document]:
    """
    Load a document from a file path using the appropriate loader.
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.pdf':
        loader = PyMuPDFLoader(path)
    elif ext == '.docx':
        loader = Docx2txtLoader(path)
    elif ext in ['.txt', '.md', '.csv']:
        loader = TextLoader(path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Supported types are .pdf, .docx, and .txt.")
        
    docs = loader.load()
    # Normalize 0-indexed page numbers to 1-indexed page numbers
    for doc in docs:
        if 'page' in doc.metadata:
            try:
                doc.metadata['page'] = int(doc.metadata['page']) + 1
            except (ValueError, TypeError):
                pass
    return docs

# Might be an issue for lengthy technical documents
def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks and append metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=['\n\n', '\n', '.', ' ']
    )
    
    chunks = splitter.split_documents(documents)
    
    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get('source', 'unknown_file')
        chunk.metadata['source_file'] = os.path.basename(source)
        chunk.metadata['chunk_index'] = idx
        
    return chunks

def ingest_documents(file_paths: List[str], persist_dir: str = './chroma_db'):
    """
    Load, chunk, and index documents into ChromaDB.
    """
    store = get_or_create_store(persist_dir)
    all_chunks = []
    
    for path in file_paths:
        try:
            docs = load_document(path)
            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)
            print(f"Indexed: {os.path.basename(path)} — {len(chunks)} chunks")
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    if all_chunks:
        add_documents(store, all_chunks)
        
    return store
