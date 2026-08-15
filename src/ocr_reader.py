"""
ocr_reader.py

Extracts text from a photo of a drug box using EasyOCR (a pretrained
text-recognition library -- no training needed on your end).

The reader is loaded once at import time, not per-request, since
loading it takes a few seconds -- doing that on every API call would
make each prediction painfully slow.
"""

import easyocr

# gpu=False since most laptops won't have a CUDA GPU -- runs on CPU,
# a bit slower per image but works everywhere without extra setup.
_reader = easyocr.Reader(["en"], gpu=False)


def extract_text_lines(image_bytes: bytes) -> list[str]:
    """
    Run OCR on raw image bytes (as received from an uploaded file)
    and return the detected text lines.
    """
    result = _reader.readtext(image_bytes, detail=0)
    return result