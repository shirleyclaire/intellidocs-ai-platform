"""Memory manager and topic switch detection for Document-Aware RAG Assistant."""
from langchain_classic.memory import ConversationBufferWindowMemory
from sentence_transformers import SentenceTransformer, util

# Module-level variable to cache the model
_ENCODER = None

def build_memory() -> ConversationBufferWindowMemory:
    """Build and return a ConversationBufferWindowMemory object."""
    # k value can be adjusted based on desired memory length.
    # Lower k -> shorter memory (less context).
    # Higher k -> longer memory (more context, more tokens used).
    return ConversationBufferWindowMemory(
        k=5,
        memory_key='history',
        return_messages=True,
        output_key='answer'
    )

def is_topic_switch(prev_query: str, new_query: str, threshold: float = 0.22) -> bool:
    """
    Check if the topic has switched significantly between queries.
    Uses sentence-transformers to encode queries and compute cosine similarity.
    """
    if not prev_query or not prev_query.strip():
        return False
        
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
    # Encode queries
    embeddings1 = _ENCODER.encode(prev_query, convert_to_tensor=True)
    embeddings2 = _ENCODER.encode(new_query, convert_to_tensor=True)
    
    # Compute cosine similarity
    cosine_score = util.cos_sim(embeddings1, embeddings2).item()

    print(f"Cosine similarity: {cosine_score}")
    
    return cosine_score < threshold
