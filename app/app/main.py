# app/main.py
import io, os
from typing import Any, Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from app.yolo_model import infer_bytes
from app.ops.color import label_colors
from app.ops.shape import label_shape
from app.ops.ocr import ocr_roi

app = FastAPI(title="Vision Service")

@app.get("/health", summary="Health")
def health():
    return {"ok": True}

def _limit_indices_by_conf(dets: List[dict], k: int) -> List[int]:
    idx = list(range(len(dets)))
    idx.sort(key=lambda i: float(dets[i]["conf"]), reverse=True)
    return idx[:k]

@app.post("/detect", summary="YOLO detect only")
async def detect(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "file must be an image")
    data = await file.read()
    return infer_bytes(data)

@app.post("/analyze", summary="Detect + color/shape/OCR")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "file must be an image")
    data = await file.read()
    det = infer_bytes(data)
    pills = []
    img = Image.open(io.BytesIO(data)).convert("RGB")

    # 정책상 최대 처리 개수 (ENV: PILL_MAX, 기본 6)
    pill_max = int(os.getenv("PILL_MAX", "6"))
    sel_idx = _limit_indices_by_conf(det.get("detections", []), pill_max)

    for i in sel_idx:
        d = det["detections"][i]
        x1,y1,x2,y2 = map(int, d["bbox"])
        # 경계 보정
        x1 = max(0, min(x1, img.width-1)); x2 = max(0, min(x2, img.width))
        y1 = max(0, min(y1, img.height-1)); y2 = max(0, min(y2, img.height))
        if x2<=x1 or y2<=y1: continue
        roi = img.crop((x1,y1,x2,y2))

        pills.append({
            "bbox": d["bbox"],
            "conf": d["conf"],
            "shape": label_shape(roi),
            "color": label_colors(roi),
            "imprint": ocr_roi(roi)
        })
    return {
        "pills": pills,
        "image": det.get("image"),
        "meta": {**det.get("meta", {}), "pill_max": pill_max}
    }
