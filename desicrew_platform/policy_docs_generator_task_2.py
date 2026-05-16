import fitz # PyMuPDF
import os

def create_pdf(filename, title, content_pages):
    doc = fitz.open()
    for page_content in content_pages:
        page = doc.new_page()
        # Draw title
        page.insert_text((50, 50), title, fontsize=20, color=(0, 0, 1))
        # Draw content
        y = 100
        for line in page_content.split('\n'):
            page.insert_text((50, y), line, fontsize=12)
            y += 20
    doc.save(filename)
    doc.close()

# Document 1: General Policy
doc1_content = [
    """Refund Policy
Customers may request a refund within 30 days of purchase.
The item must be in its original packaging and unused.
Proof of purchase is required for all refund requests.""",
    """Premium Customer Benefits
Premium customers have an extended refund window of 60 days.
They are also eligible for free return shipping on all items.
Priority support is available 24/7 for premium members."""
]

# Document 2: Claims Process
doc2_content = [
    """Claims Process
To file a claim, please visit our online portal and fill out the claim form.
You will need to provide your order number and a description of the issue.
Claims are typically processed within 5-7 business days.""",
    """Documentation for Claims
Please submit the following documents with your claim:
1. A copy of the original receipt.
2. Photos of the damaged item (if applicable).
3. A valid government-issued ID."""
]

os.makedirs("test_docs", exist_ok=True)
create_pdf("test_docs/general_policy.pdf", "General Policy", doc1_content)
create_pdf("test_docs/claims_process.pdf", "Claims Process", doc2_content)

print("Test PDFs created in test_docs/")

