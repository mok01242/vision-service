# app/ops/color.py
from typing import List
import numpy as np
from PIL import Image

PALETTE = [
    ("하양", np.array([0,0,255])), ("검정", np.array([0,0,0])), ("회색", np.array([0,0,128])),
    ("노랑", np.array([60,255,255])), ("주황", np.array([30,255,255])), ("빨강", np.array([0,255,255])),
    ("분홍", np.array([330,180,255])), ("보라", np.array([270,150,200])),
    ("파랑", np.array([210,255,255])), ("초록", np.array([120,255,255])), ("갈색", np.array([25,150,120])),
]

def label_colors(roi: Image.Image, topk: int = 2) -> List[str]:
    hsv = roi.convert("HSV")
    arr = np.array(hsv).reshape(-1,3)
    if arr.shape[0] > 50000:
        idx = np.random.choice(arr.shape[0], 50000, replace=False)
        arr = arr[idx]
    mean = arr.mean(axis=0)
    def dist(p):
        dh = min(abs(p[0]-mean[0]), 255-abs(p[0]-mean[0]))
        ds = abs(p[1]-mean[1]); dv = abs(p[2]-mean[2])
        return dh + ds*0.5 + dv*0.5
    ranked = sorted(PALETTE, key=lambda kv: dist(kv[1]))
    return [name for name,_ in ranked[:topk]]
