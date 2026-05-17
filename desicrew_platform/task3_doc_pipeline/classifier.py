import os
import json
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict
from rapidfuzz import fuzz

from .ocr_engine import OCRToken

@dataclass
class ClassificationResult:
    predicted_class: str          # e.g., "PAN", "Passport"
    fuzzy_score: float            # 0.0 to 1.0
    regex_matched: bool           # True if the class-specific regex signature was found
    flagged: bool                 # True if confidence is too low or validation failed
    flag_reason: str              # Empty string if not flagged


def fuzzy_classify(text: str, anchors: Dict[str, List[str]]) -> Tuple[str, float]:
    """
    For each class in the anchors dictionary, compute the maximum rapidfuzz.fuzz.token_set_ratio 
    score across all its anchor phrases against the concatenated input text.
    
    Returns a tuple of (best_class_name, best_score_normalized).
    """
    best_class_name = ""
    best_score = 0.0
    
    text_upper = text.upper()
    
    for class_name, anchor_phrases in anchors.items():
        for phrase in anchor_phrases:
            score = fuzz.token_set_ratio(phrase.upper(), text_upper)
            score_normalized = score / 100.0
            if score_normalized > best_score:
                best_score = score_normalized
                best_class_name = class_name
                
    return best_class_name, best_score


def validate_regex(text: str, predicted_class: str, regex_dict: Dict[str, str]) -> bool:
    """
    Fetch the regex pattern for the predicted_class from regex_dict and
    use re.search() to check if the pattern exists anywhere in the input text.
    """
    pattern = regex_dict.get(predicted_class)
    if not pattern:
        return False
    
    match = re.search(pattern, text)
    return match is not None


def classify_document(tokens: List[OCRToken], config_path: str = "config/document_classes.json") -> ClassificationResult:
    """
    Concatenate all token texts into a single uppercase string using a space delimiter.
    Load the JSON configuration containing classes, anchors, and regex patterns.
    Run classification and validation, and apply straight-through or borderline-rescue routing logic.
    """
    # 1. Concatenate all token texts into a single uppercase string
    concatenated_text = " ".join(t.text for t in tokens).upper()
    
    # 2. Load JSON config robustly, supporting path resolution relative to desicrew_platform/
    resolved_path = config_path
    if not os.path.exists(resolved_path):
        # Look relative to the parent directory of this module
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fallback_path = os.path.join(base_dir, config_path)
        if os.path.exists(fallback_path):
            resolved_path = fallback_path
            
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Could not load document classes configuration file: {e}")
        
    anchors = config.get("fuzzy_anchors", {})
    regex_dict = config.get("validation_regex", {})
    
    # 3. Fuzzy classification
    predicted_class, fuzzy_score = fuzzy_classify(concatenated_text, anchors)
    
    # 4. Regex validation
    regex_matched = validate_regex(concatenated_text, predicted_class, regex_dict)
    
    # 5. Routing Logic
    flagged = True
    flag_reason = ""
    
    # STRAIGHT_THROUGH
    if fuzzy_score >= 0.85 and regex_matched:
        flagged = False
    # BORDERLINE_RESCUE
    elif fuzzy_score >= 0.92 and not regex_matched:
        flagged = False
    # CONFLICT / FAILURE
    else:
        flag_reason = f"Low fuzzy confidence ({fuzzy_score:.4f}) or missing required regex signature."
        
    return ClassificationResult(
        predicted_class=predicted_class,
        fuzzy_score=fuzzy_score,
        regex_matched=regex_matched,
        flagged=flagged,
        flag_reason=flag_reason
    )
