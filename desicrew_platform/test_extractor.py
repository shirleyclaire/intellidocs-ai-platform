import sys
import os
import unittest

# Ensure the module can find task3_doc_pipeline relative to desicrew_platform
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from task3_doc_pipeline.ocr_engine import OCRToken
from task3_doc_pipeline.extractor import ExtractedField, regex_extract, spatial_extract, extract_fields

class TestExtractor(unittest.TestCase):
    def test_regex_extract_success(self):
        tokens = [
            OCRToken(text="My Aadhaar number is", bbox=(0, 0, 10, 10), confidence=0.9, page=0),
            OCRToken(text="1234", bbox=(0, 0, 10, 10), confidence=0.8, page=0),
            OCRToken(text="5678", bbox=(0, 0, 10, 10), confidence=0.85, page=0),
            OCRToken(text="9012", bbox=(0, 0, 10, 10), confidence=0.95, page=0),
            OCRToken(text="in this document", bbox=(0, 0, 10, 10), confidence=0.9, page=0)
        ]
        pattern = "\\b\\d{4}\\s?\\d{4}\\s?\\d{4}\\b"
        res = regex_extract("aadhaar_number", pattern, tokens)
        self.assertIsNotNone(res)
        self.assertEqual(res.value, "1234 5678 9012")
        self.assertEqual(res.ocr_confidence, 0.8)  # Minimum confidence of 1234, 5678, 9012

    def test_regex_extract_fail(self):
        tokens = [
            OCRToken(text="My Aadhaar number is 1234 5678 90", bbox=(0, 0, 10, 10), confidence=0.9, page=0)
        ]
        pattern = "\\b\\d{4}\\s?\\d{4}\\s?\\d{4}\\b"
        res = regex_extract("aadhaar_number", pattern, tokens)
        self.assertIsNone(res)

    def test_spatial_extract_right(self):
        tokens = [
            OCRToken(text="Name", bbox=(10, 100, 50, 120), confidence=0.95, page=0),
            OCRToken(text="JOHN", bbox=(60, 100, 100, 120), confidence=0.9, page=0),
            OCRToken(text="DOE", bbox=(110, 100, 150, 120), confidence=0.8, page=0),
            OCRToken(text="Random", bbox=(200, 100, 240, 120), confidence=0.9, page=0)  # out of pixel_threshold
        ]
        res = spatial_extract("full_name", "Name", tokens, direction="right", pixel_threshold=110)
        self.assertIsNotNone(res)
        self.assertEqual(res.value, "JOHN DOE")
        self.assertAlmostEqual(res.ocr_confidence, 0.85)  # Mean of 0.9 and 0.8

    def test_spatial_extract_below(self):
        tokens = [
            OCRToken(text="Government of India", bbox=(100, 10, 300, 30), confidence=0.95, page=0),
            OCRToken(text="ANIL", bbox=(110, 40, 130, 60), confidence=0.9, page=0),
            OCRToken(text="KUMAR", bbox=(140, 40, 180, 60), confidence=0.8, page=0)
        ]
        res = spatial_extract("full_name", "Government of India", tokens, direction="below", pixel_threshold=40)
        self.assertIsNotNone(res)
        self.assertEqual(res.value, "ANIL KUMAR")
        self.assertAlmostEqual(res.ocr_confidence, 0.85)

    def test_extract_fields_success(self):
        tokens = [
            OCRToken(text="Income Tax Department GOVT.OF INDIA", bbox=(10, 10, 200, 30), confidence=0.95, page=0),
            OCRToken(text="Your Name", bbox=(10, 40, 80, 55), confidence=0.99, page=0),
            OCRToken(text="ANIL KUMAR", bbox=(10, 60, 110, 75), confidence=0.9, page=0),
            OCRToken(text="Your PAN Number", bbox=(10, 90, 80, 105), confidence=0.99, page=0),
            OCRToken(text="ABCDE1234F", bbox=(10, 120, 110, 135), confidence=0.88, page=0)
        ]
        fields = extract_fields("PAN", tokens)
        self.assertEqual(len(fields), 2)
        
        # Check pan_number
        self.assertEqual(fields[0].field_name, "pan_number")
        self.assertEqual(fields[0].value, "ABCDE1234F")
        self.assertEqual(fields[0].method, "regex")
        
        # Check full_name
        self.assertEqual(fields[1].field_name, "full_name")
        self.assertEqual(fields[1].value, "ANIL KUMAR")
        self.assertEqual(fields[1].method, "spatial")

    def test_extract_fields_fallback(self):
        tokens = [
            OCRToken(text="Unrelated text", bbox=(0, 0, 10, 10), confidence=0.9, page=0)
        ]
        fields = extract_fields("PAN", tokens)
        self.assertEqual(len(fields), 2)
        
        self.assertEqual(fields[0].field_name, "pan_number")
        self.assertEqual(fields[0].value, None)
        self.assertEqual(fields[0].method, "failed")
        
        self.assertEqual(fields[1].field_name, "full_name")
        self.assertEqual(fields[1].value, None)
        self.assertEqual(fields[1].method, "failed")

if __name__ == "__main__":
    unittest.main()
