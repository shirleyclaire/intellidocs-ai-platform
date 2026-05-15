"""Vector store operations."""

from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .embeddings import get_embeddings

def get_or_create_store(persist_dir: str) -> Chroma:
    """
    Load existing Chroma DB if present, otherwise create new empty store.
    
    Args:
        persist_dir (str): Directory where the DB should be persisted.
        
    Returns:
        Chroma: Vector store instance.
    """
    embeddings = get_embeddings()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def add_documents(store: Chroma, documents: List[Document]) -> None:
    """
    Add LangChain Document objects to the store and persist changes to disk.
    
    Args:
        store (Chroma): The Chroma vector store instance.
        documents (List[Document]): The list of documents to add.
    """
    store.add_documents(documents)
    # Note: In recent versions of langchain-chroma, persistence is handled automatically 
    # based on the persist_directory provided on creation. We call persist() for backwards compatibility.
    if hasattr(store, "persist"):
        store.persist()
