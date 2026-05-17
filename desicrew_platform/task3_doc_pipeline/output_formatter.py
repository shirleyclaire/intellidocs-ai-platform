"""Output Formatter and Flagging Reporter for Document Extraction Pipeline."""
import os
import json
from typing import List, Dict, Any
from .classifier import ClassificationResult
from .extractor import ExtractedField
from .scorer import FIELD_FLAG_THRESHOLD

def format_output(
    document_id: str, 
    classification: ClassificationResult, 
    fields: List[ExtractedField]
) -> Dict[str, Any]:
    """
    Assembles the final structured extraction dictionary, evaluates field flagging thresholds,
    and constructs clear audit rationales for human-in-the-loop review queues.
    """
    flagging_rationale = []
    extraction_details = {}
    
    # 1. Evaluate Field Flagging
    any_field_flagged = False
    for field in fields:
        is_flagged = field.extraction_confidence < FIELD_FLAG_THRESHOLD
        if is_flagged:
            any_field_flagged = True
            flagging_rationale.append(
                f"SYSTEM_FLAG (Extraction): The field '{field.field_name}' requires manual review. "
                f"Extracted confidence ({field.extraction_confidence:.2f}) is below the acceptable threshold of {FIELD_FLAG_THRESHOLD}."
            )
            
        extraction_details[field.field_name] = {
            "value": field.value,
            "confidence": field.extraction_confidence,
            "method": field.method,
            "flagged": is_flagged
        }
        
    # 2. Evaluate Classification Flagging
    if classification.flagged:
        flagging_rationale.append(
            "SYSTEM_FLAG (Classification): Unable to definitively verify document type. "
            "Mandatory regex signatures were missing or anchor confidence fell below threshold."
        )
        
    # 3. Overall Document Flagging Check
    document_flagged_for_human_review = classification.flagged or any_field_flagged
    
    return {
        "document_id": document_id,
        "classification": {
            "document_type": classification.predicted_class,
            "fuzzy_match_score": float(classification.fuzzy_score),
            "regex_matched": classification.regex_matched
        },
        "extraction": extraction_details,
        "document_flagged_for_human_review": document_flagged_for_human_review,
        "flagging_rationale": flagging_rationale
    }

def write_outputs(outputs: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Writes formatted output dictionaries to individual JSON files in the target directory
    and compiles a collective `flagging_report.json` tracking all anomalies requiring human audit.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    flagging_report = {}
    
    for out in outputs:
        doc_id = out["document_id"]
        file_path = os.path.join(output_dir, f"{doc_id}.json")
        
        # Write individual document extraction record
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
            
        # Add to flagged audit report if review is needed
        if out["document_flagged_for_human_review"]:
            flagging_report[doc_id] = out["flagging_rationale"]
            
    # Write consolidated flagging report
    report_path = os.path.join(output_dir, "flagging_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(flagging_report, f, indent=2, ensure_ascii=False)
        
    print(f"[Output] Successfully wrote {len(outputs)} document records and consolidated flagging report to: '{output_dir}'")
