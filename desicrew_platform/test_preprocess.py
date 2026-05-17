import cv2
import numpy as np
import os
from task3_doc_pipeline.preprocess import preprocess_document

def main():
    # Test with a synthetic noisy image
    # Light grey background (240) to test adaptive thresholding
    test_img = np.ones((400, 800, 3), dtype=np.uint8) * 240
    
    # Add some "text"
    cv2.putText(test_img, "Test Document", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (30, 30, 30), 3)
    cv2.putText(test_img, "Handwritten Note", (100, 250), cv2.FONT_HERSHEY_COMPLEX, 1.5, (50, 50, 50), 2)
    
    # Add some noise
    noise = np.random.randint(0, 50, (400, 800, 3), dtype='uint8')
    test_img = cv2.add(test_img, noise)
    
    # Save to a temporary file because preprocess_document expects a file path
    temp_file = "temp_synthetic.png"
    cv2.imwrite(temp_file, test_img)
    
    print("Pre-processing synthetic image...")
    try:
        pages = preprocess_document(temp_file)
        if not pages:
            print("Error: No pages returned!")
            return
            
        # Convert PIL Image back to numpy array for verification
        processed = np.array(pages[0])
        
        print("Output shape:", processed.shape)
        print("Dtype:", processed.dtype)
        unique_vals = np.unique(processed)
        print("Unique values:", unique_vals)
        
        if len(unique_vals) <= 2 and (0 in unique_vals or 255 in unique_vals):
            print("Success: Image is binary (0 and 255 only).")
        else:
            print("Warning: Image is NOT strictly binary.")
            
        # Save to verify visually (convert back to BGR for cv2.imwrite)
        cv2.imwrite("test_preprocessed.png", cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        print("Saved test_preprocessed.png - check it manually")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    main()
