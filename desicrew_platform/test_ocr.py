import numpy as np
import cv2
import os
import time
from task3_doc_pipeline.ocr_engine import ocr_full_document, is_handwritten_region

def main():
    # Create a clean printed-text test image

    # path = r"C:\Users\Shirley Claire\Desktop\Shirley\Documents\ID Cards\Shirley_Aadhar_Card.pdf"

    img = r"C:\Users\Shirley Claire\Desktop\Shirley\Projects\intellidocs-ai-platform\page_0.png"

    # img = r"C:\Users\Shirley Claire\Desktop\Shirley\Projects\intellidocs-ai-platform\handwritten_text.jpeg"
    
    img = cv2.imread(img)

    start = time.time()
    result = ocr_full_document(img)
    elapsed = time.time() - start

    print(result)
    print(f"Elapsed time: {elapsed:.2f} seconds")
    # images = pdf_to_images(path)

    # cv2.imshow("Image", images[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # for i, img in enumerate(images):
    #     print(f"Processing page {i+1}")
    #     result = ocr_full_document(img)
    #     print(result)


    # img = np.ones((300, 1000, 3), dtype=np.uint8) * 255
    # cv2.putText(img, "PAN Number: ABCDE1234F", (50, 150),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    # cv2.putText(img, "Date: 2026-05-16", (50, 220),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print("Running OCR on printed text document...")
    # result = ocr_full_document(img)
    
    print("\n--- OCR Result ---")
    print("Engine used:", result['engine'])
    print("Confidence:", f"{result['confidence']:.2f}")
    print("Text extracted:")
    print("-" * 20)
    print(result['full_text'])
    print("-" * 20)
    
    if result['full_text'].strip():
        print("\nSuccess: Text extracted successfully.")
    else:
        print("\nWarning: No text extracted.")

if __name__ == "__main__":
    main()
