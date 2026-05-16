import sys
import os

workspace = r"C:\Users\Shirley Claire\Desktop\Shirley\Projects\intellidocs-ai-platform\desicrew_platform"
if workspace not in sys.path:
    sys.path.insert(0, workspace)

from task2_rag_assistant.ingest import ingest_documents

def main():
    # Create a test text file first
    with open("test_doc.txt", "w", encoding="utf-8") as f:
        f.write("""
Refund Policy
Customers may request a refund within 30 days of purchase.
Premium customers have a 60-day refund window.
Refunds are processed within 5 business days.

Claims Process
To file a claim, submit Form A with your policy number.
Claims require proof of identity and the original receipt.
Grace Period: a 15-day grace period applies to all late payments.
    """)

    store = ingest_documents(["test_doc.txt"], persist_dir="./test_chroma")

    # Verify retrieval works
    results = store.similarity_search("refund policy", k=2)
    for r in results:
        print("---")
        print("Source:", r.metadata.get("source_file"))
        print("Text:", r.page_content[:100])

if __name__ == "__main__":
    main()
