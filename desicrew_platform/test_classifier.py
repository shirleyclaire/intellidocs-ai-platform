import sys
import os
import unittest

# Ensure the module can find task3_doc_pipeline relative to desicrew_platform
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from task3_doc_pipeline.ocr_engine import OCRToken
from task3_doc_pipeline.classifier import ClassificationResult, fuzzy_classify, validate_regex, classify_document

class TestClassifier(unittest.TestCase):
    def test_fuzzy_classify(self):
        anchors = {
            "Aadhaar": ["Unique Identification Authority", "Government of India", "UIDAI"],
            "PAN": ["Income Tax Department", "Permanent Account Number"]
        }
        text = "This is a document from the Unique Identification Authority of India"
        best_class, score = fuzzy_classify(text, anchors)
        self.assertEqual(best_class, "Aadhaar")
        self.assertGreaterEqual(score, 0.8)

    def test_validate_regex_success(self):
        regex_dict = {
            "PAN": "\\b[A-Z]{5}[0-9]{4}[A-Z]{1}\\b"
        }
        text = "Permanent Account Number ABCDE1234F Income Tax Dept"
        self.assertTrue(validate_regex(text, "PAN", regex_dict))

    def test_validate_regex_fail(self):
        regex_dict = {
            "PAN": "\\b[A-Z]{5}[0-9]{4}[A-Z]{1}\\b"
        }
        text = "Permanent Account Number ABCDE12345F Income Tax Dept" # 5 digits instead of 4
        self.assertFalse(validate_regex(text, "PAN", regex_dict))

    def test_classify_document_straight_through(self):
        tokens = [
            OCRToken(text="Income Tax Department", bbox=(0, 0, 10, 10), confidence=0.99, page=0),
            OCRToken(text="Permanent Account Number", bbox=(0, 15, 10, 25), confidence=0.99, page=0),
            OCRToken(text="ABCDE1234F", bbox=(0, 30, 10, 40), confidence=0.99, page=0)
        ]
        res = classify_document(tokens)
        self.assertEqual(res.predicted_class, "PAN")
        self.assertTrue(res.regex_matched)
        self.assertFalse(res.flagged)

    def test_classify_document_borderline_rescue(self):
        tokens = [
            OCRToken(text="Income Tax Department", bbox=(0, 0, 10, 10), confidence=0.99, page=0),
            OCRToken(text="Permanent Account Number", bbox=(0, 15, 10, 25), confidence=0.99, page=0)
            # No valid regex match
        ]
        res = classify_document(tokens)
        self.assertEqual(res.predicted_class, "PAN")
        self.assertFalse(res.regex_matched)
        self.assertFalse(res.flagged) # Should trigger borderline rescue (fuzzy >= 0.92)

    def test_classify_document_flagged(self):
        tokens = [
            OCRToken(text="This is an unrelated statement of bank policy", bbox=(0, 0, 10, 10), confidence=0.99, page=0)
        ]
        res = classify_document(tokens)
        self.assertTrue(res.flagged)
        self.assertIn("Low fuzzy confidence", res.flag_reason)

if __name__ == "__main__":
    unittest.main()
