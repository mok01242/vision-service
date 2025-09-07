# app/ops/ocr.py
import os, io, json
from typing import Optional
import numpy as np
from PIL import Image

PROVIDER = os.getenv("OCR_PROVIDER", "none").lower()

def run_ocr(crop_bgr: np.ndarray) -> Optional[str]:
    """
    crop_bgr: OpenCV BGR 이미지 조각(알약 영역)
    return: 각인 텍스트(영숫자, 대문자, 최대 16자) 또는 None
    """
    if PROVIDER != "gcv":
        return None

    try:
        # 지연 임포트: GCV 켠 경우에만 로드
        from google.cloud import vision
        from google.oauth2 import service_account

        creds_json = os.getenv("GCV_CREDENTIALS_JSON")
        if not creds_json:
            return None

        info = json.loads(creds_json)
        creds = service_account.Credentials.from_service_account_info(info)
        client = vision.ImageAnnotatorClient(credentials=creds)

        # BGR -> RGB -> JPEG bytes
        img_rgb = crop_bgr[:, :, ::-1]
        pil = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        gimg = vision.Image(content=buf.getvalue())

        lang_hints = os.getenv("GCV_LANG_HINTS", "en,ko").split(",")
        resp = client.text_detection(
            image=gimg, image_context={"language_hints": lang_hints}
        )
        if resp.error.message:
            return None

        text = ""
        if resp.full_text_annotation and resp.full_text_annotation.text:
            text = resp.full_text_annotation.text

        # 각인 정제: 영숫자만, 대문자, 길이 제한
        cleaned = "".join(ch for ch in text.upper() if ch.isalnum())[:16]
        return cleaned or None
    except Exception:
        return None
