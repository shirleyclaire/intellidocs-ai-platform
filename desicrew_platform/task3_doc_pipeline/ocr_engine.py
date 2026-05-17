from dataclasses import dataclass
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
from typing import List

# Configurable constants at module level
OCR_CONFIDENCE_FLOOR = 0.40

@dataclass
class OCRToken:
    text: str
    bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    confidence: float
    page: int

# Initialise PaddleOCR once at module level (not inside the function)
# lang='en', det_model_dir=None, rec_model_dir=None to use default downloaded models.
# Since PaddleOCR >= 3.0.0 removed the show_log parameter, we use a fallback block.
try:
    ocr_model = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        det_model_dir=None,
        rec_model_dir=None,
        enable_mkldnn=False,
        show_log=False
    )
except (ValueError, TypeError):
    ocr_model = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        det_model_dir=None,
        rec_model_dir=None,
        enable_mkldnn=False
    )

def run_ocr(image: Image.Image, page_number: int = 0) -> List[OCRToken]:
    """
    Runs PaddleOCR on a preprocessed PIL Image.
    Returns a list of structured OCRToken objects.
    """
    # 1. Convert PIL Image to a numpy array
    img_array = np.array(image.convert('RGB'))

    # 2. Call ocr.ocr(img_array, cls=True) with a fallback in case cls is not supported
    try:
        result = ocr_model.ocr(img_array, cls=True)
        print('cls works bhai!')
    except TypeError:
        result = ocr_model.ocr(img_array)
        print('cls does not work bhai!')

    print('result', result)

    # 3. Process the results safely
    tokens: List[OCRToken] = []
    if not result or result[0] is None:
        return tokens

    # Get the first (and usually only) item in the result list
    data = result[0]

    # CHECK: Is this the new Dictionary format (PaddleX)?
    if isinstance(data, dict) and 'rec_texts' in data:
        # Extract the parallel lists from the dictionary keys
        texts = data.get('rec_texts', [])
        scores = data.get('rec_scores', [])
        polygons = data.get('rec_polys', data.get('dt_polys', []))

        # Loop through them by index
        for i in range(len(texts)):
            text = texts[i]
            confidence = float(scores[i])
            polygon = polygons[i]

            # Filter out low confidence tokens early
            if confidence < OCR_CONFIDENCE_FLOOR:
                continue

            # Calculate bounding box (x_min, y_min, x_max, y_max)
            x_min = int(min(pt[0] for pt in polygon))
            y_min = int(min(pt[1] for pt in polygon))
            x_max = int(max(pt[0] for pt in polygon))
            y_max = int(max(pt[1] for pt in polygon))

            tokens.append(OCRToken(
                text=text,
                bbox=(x_min, y_min, x_max, y_max),
                confidence=confidence,
                page=page_number
            ))

    # FALLBACK: The standard PaddleOCR list-of-lists format
    elif isinstance(data, list):
        for item in data:
            if not item or len(item) != 2:
                continue
                
            polygon, (text, confidence) = item
            confidence = float(confidence)

            if confidence < OCR_CONFIDENCE_FLOOR:
                continue

            x_min = int(min(pt[0] for pt in polygon))
            y_min = int(min(pt[1] for pt in polygon))
            x_max = int(max(pt[0] for pt in polygon))
            y_max = int(max(pt[1] for pt in polygon))

            tokens.append(OCRToken(
                text=text,
                bbox=(x_min, y_min, x_max, y_max),
                confidence=confidence,
                page=page_number
            ))

    return tokens

def tokens_to_text(tokens: List[OCRToken]) -> str:
    """
    Concatenate all token texts in reading order (top-to-bottom, left-to-right).
    Sort by y_min first, then x_min, with a y-grouping tolerance of 15 pixels.
    """
    if not tokens:
        return ""

    # Sort all tokens by y_min first
    sorted_tokens = sorted(tokens, key=lambda t: t.bbox[1])

    # Group tokens into lines based on y_min being within 15 pixels of the previous token
    lines: List[List[OCRToken]] = []
    current_line: List[OCRToken] = [sorted_tokens[0]]

    for token in sorted_tokens[1:]:
        prev_token = current_line[-1]
        if abs(token.bbox[1] - prev_token.bbox[1]) <= 15:
            current_line.append(token)
        else:
            lines.append(current_line)
            current_line = [token]
    lines.append(current_line)

    # Sort each line by x_min and join their texts
    line_texts: List[str] = []
    for line in lines:
        sorted_line = sorted(line, key=lambda t: t.bbox[0])
        line_text = " ".join(t.text for t in sorted_line)
        line_texts.append(line_text)

    return "\n".join(line_texts)
