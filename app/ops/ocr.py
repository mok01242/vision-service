# app/ops/ocr.py
import os, re
from typing import Optional
from PIL import Image
PROVIDER = os.getenv("OCR_PROVIDER","none")
PAT = re.compile(r"[A-Z0-9]+")

def ocr_roi(roi: Image.Image) -> Optional[str]:
    if PROVIDER == "tesseract":
        import pytesseract
        txt = pytesseract.image_to_string(
            roi, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ).upper()
        m = PAT.findall(txt)
        if not m: return None
        s = "".join(m)
        s = s.replace("O","0").replace("I","1").replace("Z","2")
        return s[:12]
    return None  # OCR 비활성화면 None
