"""Memory manager and topic switch detection for Document-Aware RAG Assistant."""
from langchain_classic.memory import ConversationBufferWindowMemory

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

def is_referential_query(query: str) -> bool:
    """
    Detects if a query is a continuation or referential follow-up that relies
    on the existing conversation context (e.g., 'explain that simply', 'why?', 'what about X?').
    """
    if not query:
        return False
        
    import re
    q = query.strip().lower()
    
    # Short queries (e.g., "why?", "how?", "yes", "no", "explain that") are highly context-dependent
    if len(q.split()) <= 4:
        return True
        
    # Keywords indicating follow-up, referential pronouns, or request for simplification/elaboration
    referential_keywords = {
        "this", "that", "it", "them", "these", "those", "above", "below", "former", "latter",
        "explain", "simplify", "simpler", "simply", "rephrase", "elaborate", "summarize", "summary",
        "more", "less", "detail", "details", "difference", "compare", "contrast", "another", "other",
        "what about", "how about", "why", "who", "when", "where", "can you", "could you", "please",
        "again", "repeat", "pardon"
    }
    
    # Check if any referential keyword is present as a standalone word in the query
    words = re.findall(r'\b\w+\b', q)
    for word in words:
        if word in referential_keywords:
            return True
            
    # Check for common multi-word referential prefixes
    for phrase in ["what about", "how about", "can you", "could you"]:
        if phrase in q:
            return True
            
    return False

def is_topic_switch(prev_query: str, new_query: str, threshold: float = 0.19) -> bool:
    """
    Check if the topic has switched significantly between queries.
    Uses sentence-transformers to encode queries and compute cosine similarity.
    Bypasses switch detection if the new query is a referential continuation of context.
    """
    if not prev_query or not prev_query.strip():
        return False
        
    # Referential/continuation queries never trigger a topic switch
    if is_referential_query(new_query):
        print(f"Referential/continuation query detected: '{new_query}'. Bypassing topic switch.")
        return False
        
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
    # Import util here as well
    from sentence_transformers import util
        
    # Encode queries
    embeddings1 = _ENCODER.encode(prev_query, convert_to_tensor=True)
    embeddings2 = _ENCODER.encode(new_query, convert_to_tensor=True)
    
    # Compute cosine similarity
    cosine_score = util.cos_sim(embeddings1, embeddings2).item()

    print(f"Cosine similarity: {cosine_score}")
    
    return cosine_score < threshold
