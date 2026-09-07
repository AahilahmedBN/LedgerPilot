"""
LedgerPilot — Phase 2: OCR Extraction (Tesseract, via pytesseract)


"""

import pytesseract


# --oem 3 = default LSTM OCR engine (Tesseract 4/5's neural engine)
# --psm 6 = "assume a single uniform block of text" — right fit for a
#           receipt/invoice after auto-crop; switch to psm 4 if you find
#           multi-column layouts (e.g. some invoice templates) OCR poorly.
TESSERACT_CONFIG = "--oem 3 --psm 6"


def extract_text(preprocessed_image):
    
    raw_text = pytesseract.image_to_string(preprocessed_image, config=TESSERACT_CONFIG)

    data = pytesseract.image_to_data(
        preprocessed_image, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DICT
    )
    confidences = [int(c) for c in data["conf"] if c not in ("-1", -1)]
    mean_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

    return raw_text.strip(), mean_confidence
