# app/ops/ocr.py
import os, json, io
from typing import Optional, Tuple
from google.cloud import vision

def _make_client() -> vision.ImageAnnotatorClient:
    sa_json = os.environ.get("GCP_VISION_CREDENTIALS_JSON")
    if not sa_json:
        raise RuntimeError("GCP_VISION_CREDENTIALS_JSON is not set")
    info = json.loads(sa_json)
    return vision.ImageAnnotatorClient.from_service_account_info(info)

def ocr_roi(image_bytes: bytes, roi: Optional[Tuple[int,int,int,int]] = None) -> str:
    """
    image_bytes: 원본 이미지 바이트
    roi: (x1, y1, x2, y2) 픽셀 좌표. 없으면 전체 영역 OCR
    """
    client = _make_client()

    # ROI 자르기(선택)
    if roi:
        from PIL import Image
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x1, y1, x2, y2 = roi
        im = im.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        content = buf.getvalue()
    else:
        content = image_bytes

    image = vision.Image(content=content)
    resp  = client.text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)

    return resp.full_text_annotation.text.strip() if resp.full_text_annotation else ""
