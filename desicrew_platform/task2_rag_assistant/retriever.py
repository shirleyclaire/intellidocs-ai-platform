"""Retrieval chain builder for Document-Aware RAG Assistant."""
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from shared.llm import get_llm
from shared.vector_store import get_or_create_store
from shared.prompts import RAG_SYSTEM_PROMPT

def build_chain(memory, persist_dir: str = './chroma_db') -> ConversationalRetrievalChain:
    """
    Build the ConversationalRetrievalChain with MMR search.
    """
    if getattr(memory, "memory_key", None) != "chat_history":
        memory.memory_key = "chat_history"

    store = get_or_create_store(persist_dir)
    
    # Create retriever with MMR
    retriever = store.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'fetch_k': 10}
    )
    
    # Build PromptTemplate
    prompt_template_str = f"""{RAG_SYSTEM_PROMPT}

Context:
{{context}}

Question:
{{question}}

Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["context", "question"]
    )
    
    # Build ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    
    return chain
