import os
import sys
from task2_rag_assistant.ingest import ingest_documents
from task2_rag_assistant.memory import build_memory, is_topic_switch
from task2_rag_assistant.retriever import build_chain

# Mock Streamlit session state for simulation
class SessionState:
    def __init__(self):
        self.history = []
        self.memory = build_memory()
        self.chain = None
        self.last_question = None

def run_turn(state, question):
    print(f"\n> User: {question}")
    
    # Check for topic switch
    if is_topic_switch(state.last_question, question):
        state.memory.clear()
        print("[INFO] Topic switched — starting fresh context.")
    
    # Invoke chain
    response = state.chain.invoke({"question": question})
    answer = response["answer"]
    sources = response.get("source_documents", [])
    
    print(f"Assistant: {answer}")
    print("Sources:")
    for doc in sources:
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "unknown")
        print(f"  Source: {source_name} - Page {page}")
    
    state.last_question = question
    return answer

def main():
    state = SessionState()
    
    # Ingest test docs
    test_docs = ["test_docs/general_policy.pdf", "test_docs/claims_process.pdf"]
    persist_dir = "./demo_chroma"
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
        
    print("Building Knowledge Base...")
    ingest_documents(test_docs, persist_dir=persist_dir)
    state.chain = build_chain(state.memory, persist_dir=persist_dir)
    
    turns = [
        "What is the refund policy?",
        "What about for premium customers?",
        "Can you explain that more simply?",
        "Is there a grace period mentioned?",
        "Compare the policies in both documents",
        "Now tell me about the claims process",
        "What documents do I need to submit?",
        "Going back to refunds — what was the window again?",
        "Summarise everything we discussed about refunds",
        "Which document section was most relevant to refunds?"
    ]
    
    for i, turn in enumerate(turns, 1):
        print(f"\n--- Turn {i} ---")
        run_turn(state, turn)

if __name__ == "__main__":
    main()
