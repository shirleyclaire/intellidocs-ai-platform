import os
import shutil
import unittest
from task3_doc_pipeline.extractor import ExtractedField
from task3_doc_pipeline.classifier import ClassificationResult
from task3_doc_pipeline.scorer import score_fields, FIELD_FLAG_THRESHOLD
from task3_doc_pipeline.output_formatter import format_output, write_outputs

class TestScorerFormatter(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./test_pipeline_outputs"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_score_fields(self):
        fields = [
            ExtractedField(field_name="F1", value="Val1", method="regex", ocr_confidence=0.9, extraction_confidence=0.0),
            ExtractedField(field_name="F2", value="Val2", method="spatial", ocr_confidence=0.8, extraction_confidence=0.0),
            ExtractedField(field_name="F3", value=None, method="failed", ocr_confidence=0.0, extraction_confidence=0.0)
        ]
        scored = score_fields(fields)
        
        # F1: regex base score 1.0 * 0.9 = 0.9
        self.assertAlmostEqual(scored[0].extraction_confidence, 0.9)
        # F2: spatial base score 0.95 * 0.8 = 0.76
        self.assertAlmostEqual(scored[1].extraction_confidence, 0.76)
        # F3: failed -> 0.0
        self.assertEqual(scored[2].extraction_confidence, 0.0)

    def test_format_output(self):
        classification = ClassificationResult(
            predicted_class="Aadhaar",
            fuzzy_score=0.95,
            regex_matched=True,
            flagged=False,
            flag_reason=""
        )
        fields = [
            ExtractedField(field_name="Aadhaar Number", value="1234 5678 9012", method="regex", ocr_confidence=0.9, extraction_confidence=0.9),
            ExtractedField(field_name="Address", value=None, method="failed", ocr_confidence=0.0, extraction_confidence=0.0)
        ]
        
        out = format_output("test-uuid-1", classification, fields)
        
        self.assertEqual(out["document_id"], "test-uuid-1")
        self.assertEqual(out["classification"]["document_type"], "Aadhaar")
        self.assertEqual(out["classification"]["fuzzy_match_score"], 0.95)
        self.assertEqual(out["classification"]["regex_matched"], True)
        
        # Check extraction details
        self.assertEqual(out["extraction"]["Aadhaar Number"]["value"], "1234 5678 9012")
        self.assertEqual(out["extraction"]["Aadhaar Number"]["flagged"], False)
        self.assertEqual(out["extraction"]["Address"]["value"], None)
        self.assertEqual(out["extraction"]["Address"]["flagged"], True)
        
        # Document should be flagged for review since Address is flagged
        self.assertTrue(out["document_flagged_for_human_review"])
        self.assertEqual(len(out["flagging_rationale"]), 1)
        self.assertIn("SYSTEM_FLAG (Extraction): The field 'Address' requires manual review", out["flagging_rationale"][0])

    def test_write_outputs(self):
        classification = ClassificationResult(
            predicted_class="Passport",
            fuzzy_score=0.90,
            regex_matched=True,
            flagged=False,
            flag_reason=""
        )
        fields_ok = [
            ExtractedField(field_name="Passport Number", value="A1234567", method="regex", ocr_confidence=0.95, extraction_confidence=0.95)
        ]
        fields_fail = [
            ExtractedField(field_name="Passport Number", value=None, method="failed", ocr_confidence=0.0, extraction_confidence=0.0)
        ]
        
        out_ok = format_output("doc-ok-1", classification, fields_ok)
        out_fail = format_output("doc-fail-1", classification, fields_fail)
        
        write_outputs([out_ok, out_fail], self.test_dir)
        
        # Verify individual files exist
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "doc-ok-1.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "doc-fail-1.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "flagging_report.json")))
        
        # Load and verify flagging report
        import json
        with open(os.path.join(self.test_dir, "flagging_report.json"), "r") as f:
            report = json.load(f)
            
        self.assertNotIn("doc-ok-1", report)
        self.assertIn("doc-fail-1", report)
        self.assertEqual(len(report["doc-fail-1"]), 1)

if __name__ == "__main__":
    unittest.main()
