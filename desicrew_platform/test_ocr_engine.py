import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
from task3_doc_pipeline.ocr_engine import run_ocr, tokens_to_text, OCRToken

class TestOCREngine(unittest.TestCase):
    @patch('task3_doc_pipeline.ocr_engine.get_ocr_model')
    def test_run_ocr_polygon_to_bbox(self, mock_get_ocr_model):
        """
        Unit test that mocks the PaddleOCR call and verifies the polygon-to-bbox
        conversion is correct for a known input polygon.
        """
        # Define a mock return value containing a known polygon
        # Points are format: [x, y]
        known_polygon = [
            [10.5, 20.3],   # pt1
            [100.1, 18.9],  # pt2
            [99.7, 50.2],   # pt3
            [11.2, 51.4]    # pt4
        ]
        
        # Mock ocr_model.ocr to return our polygon with sample text and confidence
        mock_ocr_model = MagicMock()
        mock_get_ocr_model.return_value = mock_ocr_model
        mock_ocr_model.ocr.return_value = [
            [
                [known_polygon, ("Hello World", 0.95)]
            ]
        ]
        
        # Create a blank dummy image (to pass through run_ocr)
        dummy_img = Image.new('RGB', (200, 100), color='white')
        
        # Call the run_ocr function
        tokens = run_ocr(dummy_img, page_number=3)
        
        # Assertions
        self.assertEqual(len(tokens), 1)
        token = tokens[0]
        self.assertEqual(token.text, "Hello World")
        self.assertEqual(token.confidence, 0.95)
        self.assertEqual(token.page, 3)
        
        # Verification of conversion logic:
        # x_min = int(min(pt[0] for pt in polygon)) -> int(10.5) = 10
        # y_min = int(min(pt[1] for pt in polygon)) -> int(18.9) = 18
        # x_max = int(max(pt[0] for pt in polygon)) -> int(100.1) = 100
        # y_max = int(max(pt[1] for pt in polygon)) -> int(51.4) = 51
        expected_bbox = (10, 18, 100, 51)
        self.assertEqual(token.bbox, expected_bbox)

    def test_tokens_to_text_sorting_and_grouping(self):
        """
        Verifies that tokens_to_text properly groups tokens by y_min (within a 15px tolerance)
        and sorts them horizontally by x_min.
        """
        # Line 1: y_min values around 20 (within 15px tolerance)
        t1 = OCRToken(text="World", bbox=(60, 20, 100, 30), confidence=0.9, page=0)
        t2 = OCRToken(text="Hello", bbox=(10, 18, 50, 28), confidence=0.9, page=0)
        
        # Line 2: y_min values around 55 (well outside the 15px tolerance of Line 1)
        t3 = OCRToken(text="OCR", bbox=(70, 55, 100, 65), confidence=0.9, page=0)
        t4 = OCRToken(text="Paddle", bbox=(15, 52, 60, 62), confidence=0.9, page=0)
        
        # Mix the order up when passing to helper
        unordered_tokens = [t1, t3, t2, t4]
        text_output = tokens_to_text(unordered_tokens)
        
        # Expected ordered result:
        # Line 1 sorted by x: "Hello" (10) -> "World" (60)
        # Line 2 sorted by x: "Paddle" (15) -> "OCR" (70)
        expected_text = "Hello World\nPaddle OCR"
        self.assertEqual(text_output, expected_text)

if __name__ == "__main__":
    unittest.main()
