# app/yolo_model.py
import io, os
from functools import lru_cache
from typing import Dict, Any, List
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.getenv("DETECT_MODEL_PATH", "yolov8n.pt")
CONF = float(os.getenv("DETECT_CONF", 0.25))
IOU = float(os.getenv("DETECT_IOU", 0.45))
MAX_SIZE = int(os.getenv("DETECT_MAX_SIZE", 1280))

@lru_cache(maxsize=1)
def get_model() -> YOLO:
    return YOLO(MODEL_PATH)

def infer_bytes(image_bytes: bytes) -> Dict[str, Any]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = get_model()
    results = model.predict(img, conf=CONF, iou=IOU, imgsz=MAX_SIZE, verbose=False)
    r = results[0]
    names = getattr(model.model, "names", getattr(r, "names", {}))
    dets: List[Dict[str, Any]] = []
    if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
        for b in r.boxes:
            (x1, y1, x2, y2) = [float(v) for v in b.xyxy[0].tolist()]
            conf = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())
            dets.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "class_id": cls_id,
                "class_name": (names.get(cls_id) if isinstance(names, dict) else str(cls_id))
            })
    return {
        "detections": dets,
        "image": {"width": int(r.orig_shape[1]), "height": int(r.orig_shape[0])},
        "meta": {"model": os.path.basename(MODEL_PATH), "conf": CONF, "iou": IOU},
    }
