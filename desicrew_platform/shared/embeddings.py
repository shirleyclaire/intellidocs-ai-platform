"""Embeddings utility."""

from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional

_EMBEDDINGS_INSTANCE: Optional[HuggingFaceEmbeddings] = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get or create a cached instance of HuggingFaceEmbeddings.
    
    Returns:
        HuggingFaceEmbeddings: The embeddings model instance.
    """
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _EMBEDDINGS_INSTANCE
