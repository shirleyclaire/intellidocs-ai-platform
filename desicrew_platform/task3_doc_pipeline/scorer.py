"""Field confidence scorer for Document Extraction Pipeline."""
from typing import List
from .extractor import ExtractedField

# Scoring Constants
REGEX_BASE_SCORE = 1.0
SPATIAL_BASE_SCORE = 0.95  # Spatial is inherently slightly less reliable than strict regex
FIELD_FLAG_THRESHOLD = 0.75

def score_fields(fields: List[ExtractedField]) -> List[ExtractedField]:
    """
    Calculates the extraction_confidence for each ExtractedField and updates the field in-place.
    Clamps the final score to [0.0, 1.0].
    """
    for field in fields:
        if field.value is None or field.method == "failed":
            score = 0.0
        elif field.method == "regex":
            score = REGEX_BASE_SCORE * field.ocr_confidence
        elif field.method in ("spatial", "spatial_same_token"):
            score = SPATIAL_BASE_SCORE * field.ocr_confidence
        else:
            # Keep existing confidence if it was already set (e.g. by fallback or other custom process)
            score = field.extraction_confidence
            
        # Clamp score to [0.0, 1.0]
        score = max(0.0, min(1.0, score))
        field.extraction_confidence = score
        
    return fields
