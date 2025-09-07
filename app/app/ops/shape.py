# app/ops/shape.py
from typing import Literal
import cv2, numpy as np
from PIL import Image
Shape = Literal["원형","타원","캡슐","장방형","다각형"]

def label_shape(roi: Image.Image) -> Shape:
    img = np.array(roi.convert("L"))
    img = cv2.GaussianBlur(img,(3,3),0)
    _,th = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    contours,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "다각형"
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area == 0: return "다각형"
    perimeter = cv2.arcLength(c, True)
    circularity = 4*np.pi*area/(perimeter*perimeter+1e-6)
    x,y,w,h = cv2.boundingRect(c)
    aspect = max(w,h)/max(1,min(w,h))
    if circularity>0.85: return "원형"
    if aspect>1.6 and circularity>0.65: return "캡슐"
    if aspect>1.3: return "장방형"
    return "타원"
