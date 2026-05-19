"""Embeddings utility."""

from typing import Optional, Any

_EMBEDDINGS_INSTANCE: Optional[Any] = None

def get_embeddings() -> Any:
    """
    Get or create a cached instance of HuggingFaceEmbeddings.
    
    Returns:
        HuggingFaceEmbeddings: The embeddings model instance.
    """
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _EMBEDDINGS_INSTANCE
