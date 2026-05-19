import sys
import os
import types
import json
import uuid
import tempfile
import streamlit as st

# Ensure the platform root is in the path
workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace not in sys.path:
    sys.path.insert(0, workspace)

# Dynamic mapping of real modules to the requested 'pipeline' namespace
import task3_doc_pipeline.preprocess as preprocess
import task3_doc_pipeline.ocr_engine as ocr_engine
import task3_doc_pipeline.classifier as classifier
import task3_doc_pipeline.extractor as extractor
import task3_doc_pipeline.llm_fallback as llm_fallback
import task3_doc_pipeline.scorer as scorer
import task3_doc_pipeline.output_formatter as output_formatter

# Create synthetic 'pipeline' namespace in sys.modules
pipeline_mod = types.ModuleType("pipeline")
sys.modules["pipeline"] = pipeline_mod

pipeline_mod.preprocessor = preprocess
sys.modules["pipeline.preprocessor"] = preprocess

pipeline_mod.ocr_engine = ocr_engine
sys.modules["pipeline.ocr_engine"] = ocr_engine

pipeline_mod.classifier = classifier
sys.modules["pipeline.classifier"] = classifier

pipeline_mod.extractor = extractor
sys.modules["pipeline.extractor"] = extractor

pipeline_mod.llm_fallback = llm_fallback
sys.modules["pipeline.llm_fallback"] = llm_fallback

pipeline_mod.scorer = scorer
sys.modules["pipeline.scorer"] = scorer

pipeline_mod.output_formatter = output_formatter
sys.modules["pipeline.output_formatter"] = output_formatter

# Now import using the user-specified import paths!
from pipeline.preprocessor import preprocess_document
from pipeline.ocr_engine import run_ocr, tokens_to_text
from pipeline.classifier import classify_document
from pipeline.extractor import extract_fields
from pipeline.llm_fallback import rescue_flagged_document
from pipeline.scorer import score_fields
from pipeline.output_formatter import format_output

# Page Config
st.set_page_config(layout="wide", page_title="IDP Pipeline", page_icon="🔍")

# Custom CSS for Premium UI Styling
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    h1 {
        color: #0f172a;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
    }
    .card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .field-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8em;
        font-weight: 600;
        color: white;
    }
    .stDownloadButton button {
        background-color: #0f172a;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }
    .stDownloadButton button:hover {
        background-color: #1e293b;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "pipeline_complete" not in st.session_state:
    st.session_state["pipeline_complete"] = False
if "output_json" not in st.session_state:
    st.session_state["output_json"] = {}
if "flagging_report" not in st.session_state:
    st.session_state["flagging_report"] = {}
if "pipeline_log" not in st.session_state:
    st.session_state["pipeline_log"] = []
if "ocr_text" not in st.session_state:
    st.session_state["ocr_text"] = ""

# Title & Header Block
st.title("🔍 Intelligent Document Processing (IDP) Pipeline")
st.markdown("Automated layout classification, OCR parsing, spatial field extraction, and exception handling.")

# Warning Disclaimer Banner
st.warning(
    "⚠️ **Supported Documents Disclaimer:** This system is optimized and strictly configured ONLY for the following "
    "10 specific document classes: **Aadhaar Card, PAN Card, Driving Licence, Passport, NACH Mandate, FATCA Annexure, "
    "Benefit Illustration, Moral Hazard Questionnaire, Multiple Policies Consent, Suitability Profiler**."
)

def log_message(msg: str):
    """
    Appends a formatted log message to session state, enforcing strict sensitive number redacting.
    """
    import re
    # Redact Aadhaar/12-digit patterns
    redacted = re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\b', '[Aadhaar Redacted]', msg)
    # Redact PAN Card patterns
    redacted = re.sub(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', '[PAN Redacted]', redacted)
    st.session_state["pipeline_log"].append(redacted)

# File Uploader
uploaded_file = st.file_uploader("Upload Document (Image or PDF)", type=["png", "jpg", "jpeg", "tiff", "pdf"])

process_btn = st.button("Process Document", disabled=(uploaded_file is None), use_container_width=True)

if process_btn and uploaded_file:
    # Reset State for a new run
    st.session_state["pipeline_complete"] = False
    st.session_state["output_json"] = {}
    st.session_state["flagging_report"] = {}
    st.session_state["pipeline_log"] = []
    st.session_state["ocr_text"] = ""
    
    log_message(f"Starting document pipeline processing for: '{uploaded_file.name}'")
    
    # Save uploaded file to a temporary file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name
        
    try:
        with st.status("Processing Document through IDP Pipeline...", expanded=True) as status:
            # Stage 1: Preprocessing
            status.update(label="Stage 1: Pre-processing document...")
            log_message("Pre-processing document: performing DPI conversion, deskewing, and binarisation.")
            preprocessed_images = preprocess_document(temp_path)
            if not preprocessed_images:
                raise ValueError("Pre-processing step returned an empty image array.")
            processed_pil = preprocessed_images[0]
            log_message("Document pre-processing successfully completed.")
            
            # Stage 2: PaddleOCR Extraction
            status.update(label="Stage 2: Extracting text layout...")
            log_message("Running PaddleOCR engine on preprocessed document to acquire layout coordinates.")
            tokens = run_ocr(processed_pil)
            st.session_state["ocr_text"] = tokens_to_text(tokens)
            log_message(f"OCR Complete. Detected {len(tokens)} text coordinate tokens.")
            
            # Stage 3: Hybrid Classification
            status.update(label="Stage 3: Classifying document type...")
            log_message("Evaluating document type using hybrid fuzzy anchor weights and regular expressions.")
            classification_res = classify_document(tokens)
            log_message(
                f"Classification Result: predicted class '{classification_res.predicted_class}' "
                f"with fuzzy match score {classification_res.fuzzy_score:.2f}."
            )
            
            # Stage 4: Deterministic Field Extraction
            status.update(label="Stage 4: Performing layout field extraction...")
            log_message(f"Extracting target schema fields for '{classification_res.predicted_class}' using spatial heuristics.")
            extracted_fields = extract_fields(classification_res.predicted_class, tokens)
            log_message(f"Extracted {len(extracted_fields)} schema fields from target layout.")
            
            # Stage 5: Confidence Scoring
            status.update(label="Stage 5: Scoring extraction confidence...")
            log_message("Scoring extraction results based on parser methodology and token confidence.")
            scored_fields = score_fields(extracted_fields)
            log_message("Deterministic field scoring completed.")
            
            # Stage 6: Exception Fallback Rescue
            status.update(label="Stage 6: Checking exception boundaries...")
            log_message("Reviewing field confidence scores for exception routing / Gemini fallbacks.")
            rescued_fields = rescue_flagged_document(processed_pil, classification_res.predicted_class, scored_fields)
            final_scored_fields = score_fields(rescued_fields)
            log_message("Exception rescue checks completed.")
            
            # Stage 7: Final Format & Serialization
            status.update(label="Stage 7: Writing final structured output...")
            log_message("Assembling final compliant structured JSON output and flagging reports.")
            doc_id = str(uuid.uuid4())
            output_json = format_output(doc_id, classification_res, final_scored_fields)
            
            # Store in Session State
            st.session_state["output_json"] = output_json
            if output_json["document_flagged_for_human_review"]:
                st.session_state["flagging_report"] = {
                    doc_id: output_json["flagging_rationale"]
                }
                log_message("Document successfully processed. Flags detected: Routed to manual review queue.")
            else:
                st.session_state["flagging_report"] = {}
                log_message("Document successfully processed with straight-through confidence.")
                
            st.session_state["pipeline_complete"] = True
            status.update(label="Processing Complete!", state="complete", expanded=False)
            
    except Exception as e:
        log_message(f"[ERROR] Pipeline execution crashed: {str(e)}")
        st.error(f"❌ **Pipeline Error:** {str(e)}")
        
    finally:
        # Securely delete the temporary file from the disk
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as clean_err:
                print(f"Error cleaning up temp file: {clean_err}")

# Live Execution Log Panel
if st.session_state["pipeline_log"]:
    with st.expander("📝 Live Execution Logs", expanded=not st.session_state["pipeline_complete"]):
        for log in st.session_state["pipeline_log"]:
            st.code(log, language="bash")
            
        if st.session_state.get("ocr_text"):
            if st.toggle("🔍 Show Raw OCR Extracted Text"):
                st.text_area("OCR Output", st.session_state["ocr_text"], height=200)

# Results Presentation Column Layout
if st.session_state["pipeline_complete"]:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Extraction Results")
        st.json(st.session_state["output_json"])
        
    with col2:
        st.subheader("🏷️ Classification & Flags")
        out = st.session_state["output_json"]
        
        # Predicted Class Badge and metrics
        st.markdown(
            f'<div style="margin-bottom: 20px;">'
            f'<span style="background-color: #1e3a8a; color: white; padding: 6px 14px; '
            f'border-radius: 8px; font-weight: 700; font-size: 1.1em; letter-spacing: 0.5px;">'
            f'{out["classification"]["document_type"]}'
            f'</span></div>',
            unsafe_allow_html=True
        )
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(
                label="Fuzzy Match Score",
                value=f"{out['classification']['fuzzy_match_score']:.2%}"
            )
        with m_col2:
            st.metric(
                label="Regex Matched",
                value="Yes" if out['classification']['regex_matched'] else "No"
            )
            
        # Review Flag Status Banners
        if out["document_flagged_for_human_review"]:
            st.error("⚠️ **Manual Review Required:** This document has been flagged for human auditing.")
            for reason in out["flagging_rationale"]:
                st.markdown(f"- 🚩 *{reason}*")
        else:
            st.success("✅ **Straight-Through Processing:** This document passed all verification thresholds.")
            
        # Display Extracted Fields List
        st.markdown("### Extracted Schema Fields")
        for field_name, info in out["extraction"].items():
            conf = info["confidence"]
            val = info["value"]
            method = info["method"]
            flagged = info.get("flagged", False)
            
            # Use three columns for perfect vertical alignment and scannability
            col_key, col_val, col_badge = st.columns([3, 5, 2])
            
            with col_key:
                flag_tag = ' <span style="background:#7f1d1d;color:#fca5a5;padding:2px 6px;border-radius:4px;font-size:0.75em;margin-left:6px">REVIEW</span>' if flagged else ""
                st.markdown(f'<div style="padding-top: 4px;"><span style="color:#94a3b8; font-weight:600; font-size:0.95em;">{field_name}</span>{flag_tag}</div>', unsafe_allow_html=True)
                
            with col_val:
                if val is None or str(val).strip() == "":
                    st.markdown('<div style="padding-top: 4px;"><span style="color:#fbbf24; font-style:italic; font-weight:bold; font-size:0.95em;">—</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="padding-top: 4px;"><span style="color:#e2e8f0; font-weight:500; font-size:0.95em;">{val}</span></div>', unsafe_allow_html=True)
                    
            with col_badge:
                if conf >= 0.85:
                    badge_style = "background:#065f46; color:#34d399; padding:3px 8px; border-radius:6px; font-size:0.8em; font-weight:600; display:inline-block;"
                elif conf >= 0.75:
                    badge_style = "background:#78350f; color:#fbbf24; padding:3px 8px; border-radius:6px; font-size:0.8em; font-weight:600; display:inline-block;"
                else:
                    badge_style = "background:#7f1d1d; color:#fca5a5; padding:3px 8px; border-radius:6px; font-size:0.8em; font-weight:600; display:inline-block;"
                
                st.markdown(f'<div style="text-align: right;"><span style="{badge_style}">{conf:.1%}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="text-align: right; font-size: 0.75rem; color: #64748b; margin-top: 2px;">Method: {method}</div>', unsafe_allow_html=True)
                
            st.markdown("<hr style='margin:4px 0; border-color:#1e293b;'>", unsafe_allow_html=True)
            
    # Downloads Section
    st.markdown("---")
    st.subheader("📥 Download Pipeline Outputs")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        st.download_button(
            label="💾 Download output.json",
            data=json.dumps(st.session_state["output_json"], indent=2, ensure_ascii=False),
            file_name="output.json",
            mime="application/json",
            use_container_width=True
        )
        
    with d_col2:
        st.download_button(
            label="📋 Download flagging_report.json",
            data=json.dumps(st.session_state["flagging_report"], indent=2, ensure_ascii=False),
            file_name="flagging_report.json",
            mime="application/json",
            use_container_width=True
        )
