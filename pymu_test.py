import pymupdf  # PyMuPDF

def pdf_to_images(pdf_path):
    # Open the PDF document
    doc = pymupdf.open(pdf_path)

    for i, page in enumerate(doc):
        # Render page to a pixmap (image)
        pix = page.get_pixmap()
        # Save the image as a PNG
        pix.save(f"page_{i}.png")

    doc.close()

pdf_to_images(r"C:\Users\Shirley Claire\Desktop\Shirley\Documents\ID Cards\Shirley_Aadhar_Card.pdf")
