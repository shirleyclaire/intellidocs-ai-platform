"""Gemini Multimodal LLM Fallback (Exception Handling) for Document Extraction Pipeline."""
import os
import json
import tomllib
from typing import List
from PIL import Image
import google.generativeai as genai

from .extractor import ExtractedField
from .scorer import FIELD_FLAG_THRESHOLD

def load_gemini_api_key() -> str:
    """
    Robust API Key loader. Looks in:
    1. GEMINI_API_KEY environment variable.
    2. .streamlit/secrets.toml under [gemini] api_key.
    """
    # 1. Environment variable check
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()
        
    # 2. Check streamlit secrets.toml in multiple levels
    possible_paths = [
        os.path.join(".streamlit", "secrets.toml"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".streamlit", "secrets.toml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".streamlit", "secrets.toml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".streamlit", "secrets.toml"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                    key = data.get("gemini", {}).get("api_key")
                    if key:
                        return key.strip()
            except Exception as e:
                print(f"[Warning] Error reading streamlit secrets at {path}: {e}")
                
    # If no key is configured, return empty string so it fails gracefully on actual API calls
    return ""

def rescue_flagged_document(
    image: Image.Image, 
    doc_class: str, 
    fields: List[ExtractedField]
) -> List[ExtractedField]:
    """
    Intercepts documents with missing or low-confidence fields and passes
    the preprocessed image directly to Gemini 1.5 Flash to rescue those fields.
    """
    # 1. Identify missing or low-confidence fields
    missing_fields = []
    field_map = {}
    
    for field in fields:
        is_missing = field.value is None or field.method == "failed"
        is_low_conf = field.extraction_confidence < FIELD_FLAG_THRESHOLD
        
        if is_missing or is_low_conf:
            missing_fields.append(field.field_name)
            field_map[field.field_name] = field
            
    # If all fields are high confidence, return immediately
    if not missing_fields:
        return fields
        
    # 2. Load API key and initialize genai
    api_key = load_gemini_api_key()
    if not api_key:
        print("[Warning] Gemini API key is missing. Skipping LLM Fallback rescue.")
        return fields
        
    try:
        genai.configure(api_key=api_key)
        
        # 3. Construct prompt
        missing_str = ", ".join(f"'{name}'" for name in missing_fields)
        prompt = (
            f"You are an IDP extraction assistant. Analyze the provided {doc_class} document.\n"
            f"Extract the following missing or low-confidence fields: {missing_str}.\n"
            f"Return ONLY a valid JSON object where keys are the exact field names and values are the extracted strings.\n"
            f"If you cannot find a value for a field, return null."
        )
        
        # 4. Generate content
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            [prompt, image],
            generation_config={"response_mime_type": "application/json"}
        )
        
        # 5. Parse response
        if not response or not response.text:
            print("[Warning] Empty response received from Gemini.")
            return fields
            
        parsed_data = json.loads(response.text)
        
        # 6. Update ExtractedField objects in place
        for field_name, new_val in parsed_data.items():
            if field_name in field_map and new_val is not None:
                field = field_map[field_name]
                field.value = str(new_val)
                field.method = "llm_fallback"
                field.extraction_confidence = 0.90
                print(f"[Rescue] Successfully rescued '{field_name}' with value '{new_val}'")
                
    except Exception as e:
        print(f"[Warning] Failed to rescue document fields via Gemini fallback: {e}")
        
    return fields
