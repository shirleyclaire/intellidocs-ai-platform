import os
import cv2
import numpy as np
import uuid
import json
from task3_doc_pipeline.preprocess import preprocess_document
from task3_doc_pipeline.ocr_engine import run_ocr, tokens_to_text
from task3_doc_pipeline.classifier import classify_document
from task3_doc_pipeline.extractor import extract_fields
from task3_doc_pipeline.scorer import score_fields
from task3_doc_pipeline.llm_fallback import rescue_flagged_document
from task3_doc_pipeline.output_formatter import format_output, write_outputs


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
    import sys
    headless = "--headless" in sys.argv
    
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
    
    all_records = []
    
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
            
            # Run the hybrid document classifier
            print(f"Running hybrid classifier on {label}...")
            classification_res = classify_document(tokens)
            
            # Print the classification result
            print(f"\n--- Classification Result for {label} ---")
            print(f"Predicted Class: {classification_res.predicted_class}")
            print(f"Fuzzy Score:     {classification_res.fuzzy_score:.4f}")
            print(f"Regex Matched:   {classification_res.regex_matched}")
            print(f"Flagged:         {classification_res.flagged}")
            if classification_res.flagged:
                print(f"Flag Reason:     {classification_res.flag_reason}")
            print("-" * 40 + "\n")
            
            # Run the field extractor
            print(f"Running field extractor on {label}...")
            extracted_fields = extract_fields(classification_res.predicted_class, tokens)
            
            # Run the field confidence scorer
            print(f"Scoring deterministic fields on {label}...")
            scored_fields = score_fields(extracted_fields)
            
            # Print deterministic scores
            print("\n--- Deterministic Extraction Scores ---")
            for field in scored_fields:
                print(f"Field: {field.field_name:<25} | Method: {field.method:<8} | OCR Conf: {field.ocr_confidence:.4f} | Extr Conf: {field.extraction_confidence:.4f}")
            print("-" * 50 + "\n")
            
            # Trigger LLM Fallback Multimodal Exception Handling for low confidence / missing fields
            print(f"Triggering LLM Fallback (Gemini Multimodal Exception Handling) for {label}...")
            rescued_fields = rescue_flagged_document(processed_pil, classification_res.predicted_class, scored_fields)
            
            # Re-score rescued fields to finalize extraction confidence
            final_scored_fields = score_fields(rescued_fields)
            
            # Print final scores
            print("\n--- Final Post-Rescue Scores ---")
            for field in final_scored_fields:
                print(f"Field: {field.field_name:<25} | Method: {field.method:<12} | Extr Conf: {field.extraction_confidence:.4f} | Value: {str(field.value):<30}")
            print("-" * 50 + "\n")
            
            # Assemble and Format Final Compliant JSON Output
            doc_uuid = str(uuid.uuid4())
            final_record = format_output(doc_uuid, classification_res, final_scored_fields)
            
            # Print formatted output JSON
            print("\n--- Structured Extraction output.json Record ---")
            print(json.dumps(final_record, indent=2, ensure_ascii=False))
            print("-" * 60 + "\n")
            
            all_records.append(final_record)
            
            if not headless:
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
            else:
                print(f"[Headless] Skipped graphical display window for {label}.")
            
        except Exception as e:
            print(f"Error processing {label} from folder '{folder_name}': {e}")
            
    # Write final outputs and flagging report
    output_directory = "./task3_pipeline_outputs"
    print(f"\nWriting final outputs to '{output_directory}'...")
    try:
        write_outputs(all_records, output_directory)
    except Exception as e:
        print(f"Error writing final pipeline outputs: {e}")
        
    print("\n" + "=" * 60)
    print("Task 3 Preprocessing OCR Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
