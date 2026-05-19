import os
import re
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from rapidfuzz import fuzz

from .ocr_engine import OCRToken

@dataclass
class ExtractedField:
    field_name: str
    value: str | None             # None if not found
    method: str                   # "regex" or "spatial" or "failed"
    ocr_confidence: float         # Minimum confidence of the OCR token(s) used
    extraction_confidence: float  # Set to 0.0 here (Calculated later in scorer.py)


def regex_extract(field_name: str, pattern: str, tokens: List[OCRToken]) -> Optional[ExtractedField]:
    """
    Concatenate all token texts while keeping a mapping of which character range
    corresponds to which OCRToken. Run re.search() on the concatenated text and
    return an ExtractedField with the minimum OCR confidence of all matched tokens.
    """
    if not pattern or not tokens:
        return None
        
    concatenated_text = ""
    # Map each index in concatenated_text to the corresponding OCRToken
    char_mapping = []
    
    for i, token in enumerate(tokens):
        text = token.text
        if i > 0:
            concatenated_text += " "
            char_mapping.append(None)  # Space maps to no token
            
        start_idx = len(concatenated_text)
        concatenated_text += text
        for _ in range(len(text)):
            char_mapping.append(token)
            
    match = re.search(pattern, concatenated_text, re.IGNORECASE)
    if not match:
        return None
        
    start_char, end_char = match.span()
    
    # Identify exactly which tokens make up the matched substring
    matched_tokens = []
    seen_ids = set()
    
    for idx in range(start_char, end_char):
        token = char_mapping[idx]
        if token is not None and id(token) not in seen_ids:
            seen_ids.add(id(token))
            matched_tokens.append(token)
            
    if not matched_tokens:
        min_conf = 0.0
    else:
        min_conf = min(t.confidence for t in matched_tokens)
        
    return ExtractedField(
        field_name=field_name,
        value=match.group(0),
        method="regex",
        ocr_confidence=float(min_conf),
        extraction_confidence=0.0
    )


def spatial_extract(
    field_name: str, 
    anchor_phrase: str, 
    tokens: List[OCRToken], 
    direction: str = "right", 
    pixel_threshold: int = 100
) -> Optional[ExtractedField]:
    """
    Finds the token matching the anchor phrase (fuzzy ratio >= 75).
    Collects adjacent/underneath tokens within pixel_threshold based on direction.
    """
    if direction not in ("right", "below"):
        raise ValueError("direction must be strictly 'right' or 'below'.")
        
    if not tokens or not anchor_phrase:
        return None
        
    # 1. Anchor Search
    best_anchor = None
    best_score = 0.0
    
    for token in tokens:
        r_score = fuzz.ratio(anchor_phrase.upper(), token.text.upper())
        p_score = fuzz.partial_ratio(anchor_phrase.upper(), token.text.upper())
        # If the token is long and contains the anchor, partial_ratio will be high.
        # We penalize partial_ratio slightly (e.g., -5) so exact matches still win if there's a tie.
        score = max(r_score, p_score - 5 if len(token.text) > len(anchor_phrase) + 2 else 0)
        
        if score >= 75 and score > best_score:
            best_score = score
            best_anchor = token
            
    if not best_anchor:
        return None
        
    # 1.5 Same-token extraction fallback
    # If the OCR engine merged the key and value into one token (e.g. "FATHER'S NAME S/O KUMAR"),
    # the anchor token text will be significantly longer than the anchor phrase.
    if len(best_anchor.text) > len(anchor_phrase) + 3:
        # Use regex to gently strip out the anchor phrase and any immediate punctuation (*, -, :, spaces)
        pattern = re.compile(f"{re.escape(anchor_phrase)}[\\s\\*\\-:]*", re.IGNORECASE)
        match = pattern.search(best_anchor.text)
        if match:
            remaining_text = best_anchor.text[match.end():].strip()
            # Stop if we hit a slash or pipe which often separates next fields on the same line
            remaining_text = re.split(r"[/|]", remaining_text)[0].strip()
            if remaining_text and len(remaining_text) > 2:
                return ExtractedField(
                    field_name=field_name,
                    value=remaining_text,
                    method="spatial_same_token",
                    ocr_confidence=float(best_anchor.confidence),
                    extraction_confidence=0.0
                )

    # Unpack anchor coordinates
    ax_min, ay_min, ax_max, ay_max = best_anchor.bbox
    
    collected_tokens = []
    
    # 2. Spatial Collection Logic
    for token in tokens:
        if token is best_anchor:
            continue
            
        tx_min, ty_min, tx_max, ty_max = token.bbox
        
        if direction == "right":
            # Right-direction constraints:
            # - x_min > anchor.x_max
            # - abs(y_min - anchor.y_min) <= 20px (same line)
            # - distance <= pixel_threshold
            if tx_min > ax_max and abs(ty_min - ay_min) <= 20 and (tx_min - ax_max) <= pixel_threshold:
                collected_tokens.append(token)
                
        elif direction == "below":
            # Below-direction constraints:
            # - y_min > anchor.y_max
            # - distance <= pixel_threshold
            # - x-axis constraint: abs(x_min - anchor.x_min) <= 50px (directly underneath)
            if ty_min > ay_max and (ty_min - ay_max) <= pixel_threshold and abs(tx_min - ax_min) <= 50:
                collected_tokens.append(token)
                
    if not collected_tokens:
        return None
        
    # 3. Sort collected tokens in reading order
    if direction == "right":
        collected_tokens.sort(key=lambda t: t.bbox[0])  # Sort horizontally by x_min
    else:
        # Sort candidates vertically by y_min to find the closest one underneath
        collected_tokens.sort(key=lambda t: t.bbox[1])
        first_y = collected_tokens[0].bbox[1]
        # Keep only tokens on the same line as the closest underneath candidate (tolerance: 20px)
        collected_tokens = [t for t in collected_tokens if abs(t.bbox[1] - first_y) <= 20]
        # Sort horizontally by x_min for natural left-to-right reading order representation
        collected_tokens.sort(key=lambda t: t.bbox[0])
        
    joined_text = " ".join(t.text for t in collected_tokens)
    mean_conf = sum(t.confidence for t in collected_tokens) / len(collected_tokens)
    
    return ExtractedField(
        field_name=field_name,
        value=joined_text,
        method="spatial",
        ocr_confidence=float(mean_conf),
        extraction_confidence=0.0
    )


def extract_fields(
    doc_class: str, 
    tokens: List[OCRToken], 
    config_path: str = "config/document_classes.json"
) -> List[ExtractedField]:
    """
    Loads configuration patterns, looks up the extraction plan for the given class,
    and returns a list of ExtractedField items. Fills in default failed fields for fallbacks.
    """
    # Load JSON config robustly
    resolved_path = config_path
    if not os.path.exists(resolved_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fallback_path = os.path.join(base_dir, config_path)
        if os.path.exists(fallback_path):
            resolved_path = fallback_path
            
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Could not load document classes configuration file: {e}")
        
    regex_patterns = config.get("field_extraction_regex", {})
    
    # Define EXTRACTION_PLAN mapping each of the 10 classes to expected fields and functions
    EXTRACTION_PLAN = {
        "Aadhaar": [
            ("Aadhaar Number", lambda t: regex_extract("Aadhaar Number", regex_patterns.get("aadhaar_number", ""), t)),
            ("Full Name", lambda t: spatial_extract("Full Name", "Name", t, direction="below", pixel_threshold=80)),
            ("Date of Birth", lambda t: regex_extract("Date of Birth", regex_patterns.get("date_dmy", ""), t)),
            ("Address", lambda t: spatial_extract("Address", "Address", t, direction="right", pixel_threshold=250))
        ],
        "PAN": [
            ("PAN Number", lambda t: regex_extract("PAN Number", regex_patterns.get("pan_number", ""), t)),
            ("Full Name", lambda t: spatial_extract("Full Name", "Name", t, direction="below", pixel_threshold=80)),
            ("Father's Name", lambda t: spatial_extract("Father's Name", "Father's Name", t, direction="below", pixel_threshold=80)),
            ("Date of Birth", lambda t: regex_extract("Date of Birth", regex_patterns.get("date_dmy", ""), t))
        ],
        "DrivingLicence": [
            ("DL Number", lambda t: regex_extract("DL Number", regex_patterns.get("dl_number", ""), t)),
            ("Name", lambda t: spatial_extract("Name", "Name", t, direction="right", pixel_threshold=150)),
            ("Date of Issue", lambda t: regex_extract("Date of Issue", regex_patterns.get("date_dmy", ""), t)),
            ("Valid Till date", lambda t: regex_extract("Valid Till date", regex_patterns.get("date_dmy", ""), t)) # Grabs the 2nd date; will need tuning if it grabs issue date again
        ],
        "Passport": [
            ("Passport Number", lambda t: regex_extract("Passport Number", regex_patterns.get("passport_number", ""), t)),
            ("Date of Birth", lambda t: regex_extract("Date of Birth", regex_patterns.get("date_dmy", ""), t)),
            ("Date of Expiry", lambda t: regex_extract("Date of Expiry", regex_patterns.get("date_dmy", ""), t)),
            ("MRZ Line 2", lambda t: regex_extract("MRZ Line 2", regex_patterns.get("mrz_line_2", ""), t))
        ],
        "NACH": [
            ("Bank Account Number", lambda t: regex_extract("Bank Account Number", regex_patterns.get("bank_account_number", ""), t)),
            ("IFSC Code", lambda t: regex_extract("IFSC Code", regex_patterns.get("ifsc_code", ""), t)),
            ("Bank Name", lambda t: spatial_extract("Bank Name", "Bank", t, direction="right", pixel_threshold=200)),
            ("Amount (figures)", lambda t: regex_extract("Amount (figures)", regex_patterns.get("amount_figures", ""), t)),
            ("Frequency", lambda t: spatial_extract("Frequency", "Frequency", t, direction="right", pixel_threshold=100))
        ],
        "FATCA": [
            ("Policy Number", lambda t: spatial_extract("Policy Number", "Policy No", t, direction="right", pixel_threshold=150) or regex_extract("Policy Number", regex_patterns.get("application_number", ""), t)),
            ("TIN / PAN", lambda t: regex_extract("TIN / PAN", regex_patterns.get("pan_number", ""), t) or regex_extract("TIN / PAN", r"\b[A-Z]{5}[0-9]{3}[A-Z0-9]{2}\b", t)),
            ("Father's Name", lambda t: spatial_extract("Father's Name", "Father", t, direction="right", pixel_threshold=150)),
            ("Place of Birth", lambda t: spatial_extract("Place of Birth", "Place of Birth", t, direction="right", pixel_threshold=150)),
            ("Nationality", lambda t: spatial_extract("Nationality", "Nationality", t, direction="right", pixel_threshold=100))
        ],
        "BenefitIllustration": [
            ("Application Number", lambda t: regex_extract("Application Number", regex_patterns.get("application_number", ""), t)),
            ("Policyholder Name", lambda t: spatial_extract("Policyholder Name", "Proposer", t, direction="right", pixel_threshold=200)),
            ("Date", lambda t: regex_extract("Date", regex_patterns.get("date_dmy", ""), t)),
            ("Place", lambda t: spatial_extract("Place", "Place", t, direction="right", pixel_threshold=100))
        ],
        "MoralHazard": [
            ("Application Number", lambda t: regex_extract("Application Number", regex_patterns.get("application_number", ""), t)),
            ("Name of Life Assured", lambda t: spatial_extract("Name of Life Assured", "Life Assured", t, direction="right", pixel_threshold=200)),
            ("Nominee Relationship", lambda t: spatial_extract("Nominee Relationship", "Relationship", t, direction="right", pixel_threshold=150)),
            ("Date", lambda t: regex_extract("Date", regex_patterns.get("date_dmy", ""), t)),
            ("Place", lambda t: spatial_extract("Place", "Place", t, direction="right", pixel_threshold=100))
        ],
        "MultiplePolicies": [
            ("Proposer Name", lambda t: spatial_extract("Proposer Name", "Proposer", t, direction="right", pixel_threshold=200)),
            ("Reason for Multiple Policies", lambda t: spatial_extract("Reason for Multiple Policies", "Reason", t, direction="below", pixel_threshold=150)),
            ("Date", lambda t: regex_extract("Date", regex_patterns.get("date_dmy", ""), t)),
            ("Place", lambda t: spatial_extract("Place", "Place", t, direction="right", pixel_threshold=100))
        ],
        "SuitabilityProfiler": [
            ("Application Number", lambda t: regex_extract("Application Number", regex_patterns.get("application_number", ""), t)),
            ("Name of Life Assured", lambda t: spatial_extract("Name of Life Assured", "Life Assured", t, direction="right", pixel_threshold=200)),
            ("Name of Agent/SP", lambda t: spatial_extract("Name of Agent/SP", "Agent", t, direction="right", pixel_threshold=200)),
            ("Date", lambda t: regex_extract("Date", regex_patterns.get("date_dmy", ""), t)),
            ("Place", lambda t: spatial_extract("Place", "Place", t, direction="right", pixel_threshold=100))
        ]
    }
    
    plan = EXTRACTION_PLAN.get(doc_class, [])
    results = []
    
    for expected_name, extractor_fn in plan:
        try:
            field_res = extractor_fn(tokens)
        except Exception:
            field_res = None
            
        if field_res is None:
            results.append(ExtractedField(
                field_name=expected_name,
                value=None,
                method="failed",
                ocr_confidence=0.0,
                extraction_confidence=0.0
            ))
        else:
            results.append(field_res)
            
    return results
