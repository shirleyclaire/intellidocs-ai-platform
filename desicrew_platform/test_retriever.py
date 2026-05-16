import sys
import os

workspace = r"C:\Users\Shirley Claire\Desktop\Shirley\Projects\intellidocs-ai-platform\desicrew_platform"
if workspace not in sys.path:
    sys.path.insert(0, workspace)

from task2_rag_assistant.memory import build_memory, is_topic_switch
from task2_rag_assistant.retriever import build_chain

def main():
    memory = build_memory()
    chain = build_chain(memory, persist_dir="./test_chroma")

    # Turn 1
    r1 = chain.invoke({"question": "What is the refund policy?"})
    print("Answer 1:", r1['answer'])
    print("Sources:", [d.metadata.get('source_file') for d in r1['source_documents']])

    # Turn 2 — same topic, should NOT trigger topic switch
    print("Topic switch (should be False):", is_topic_switch("What is the refund policy?", "How long is the refund window?"))

    # Turn 3 — different topic, should trigger topic switch
    print("Topic switch (should be True):", is_topic_switch("What is the refund policy?", "How do I file a claim?"))

if __name__ == "__main__":
    main()
