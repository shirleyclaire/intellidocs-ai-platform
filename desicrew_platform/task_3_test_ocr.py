import os
import cv2
import numpy as np
from task3_doc_pipeline.preprocess import preprocess_document
from task3_doc_pipeline.ocr_engine import run_ocr, tokens_to_text

def find_first_image(folder_path: str) -> str:
    """Finds the first valid image file in the specified directory."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")
        
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.tif')
    for f in os.listdir(folder_path):
        if f.lower().endswith(valid_extensions):
            return os.path.join(folder_path, f)
            
    raise FileNotFoundError(f"No valid image files found in: {folder_path}")

def main():
    # Base path for the task_3_datasets folder
    # Located in the parent directory of desicrew_platform
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "task_3_datasets"))
    
    subfolders = [
        ("Aadhar Card", "aadhar_card_dataset"),
        ("Driving License", "driving_license_dataset"),
        ("PAN Card", "pan_card_dataset"),
        ("Passport", "passport_dataset")
    ]
    
    print("=" * 60)
    print("Starting Task 3 Document Preprocessing OCR Test")
    print("=" * 60)
    
    for label, folder_name in subfolders:
        folder_path = os.path.join(base_dir, folder_name)
        try:
            image_path = find_first_image(folder_path)
            print(f"\nProcessing {label} using file: {os.path.basename(image_path)}")
            
            # Preprocess the document using the pipeline
            preprocessed_images = preprocess_document(image_path)
            
            if not preprocessed_images:
                print(f"Warning: Preprocessing returned no pages for {label}")
                continue

            cv2.imshow("Original Image", cv2.imread(image_path))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                
            # Take the first page
            processed_pil = preprocessed_images[0]
            
            # Run OCR on the preprocessed image
            print(f"Running PaddleOCR on preprocessed {label}...")
            tokens = run_ocr(processed_pil)
            full_text = tokens_to_text(tokens)
            
            # Print the extracted text output
            print(f"\n--- Extracted Text for {label} ---")
            print(full_text if full_text.strip() else "[No text detected with high confidence]")
            print("-" * 40 + "\n")
            
            # Convert PIL image (RGB) to OpenCV format (BGR) for correct visualization
            processed_cv = cv2.cvtColor(np.array(processed_pil), cv2.COLOR_RGB2BGR)
            
            # Show the output using cv2.imshow
            window_name = f"Preprocessed {label} - Press any key to continue"
            print(f"Displaying window: '{window_name}'")
            
            # Resize for display if the image is too large for the screen
            h, w = processed_cv.shape[:2]
            max_height = 800
            if h > max_height:
                scale = max_height / h
                new_w = int(w * scale)
                display_img = cv2.resize(processed_cv, (new_w, max_height), interpolation=cv2.INTER_AREA)
            else:
                display_img = processed_cv
                
            cv2.imshow(window_name, display_img)
            print("Please click on the image window and press any key to proceed to the next document...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error processing {label} from folder '{folder_name}': {e}")
            
    print("\n" + "=" * 60)
    print("Task 3 Preprocessing OCR Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
