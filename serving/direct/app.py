"""FastAPI-direct: the model served IN the web process — the head-to-head baseline.

Same API as the gateway (POST /detect -> JSON), but onnxruntime runs inside
FastAPI: no dynamic batching, no instance management, every request is batch-1.
This is the '20 lines and it works' architecture — we benchmark it honestly
against Triton under concurrent load.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import decode, preprocess

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.onnx")

app = FastAPI(title="PPE Detection Direct")
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)


@app.get("/health")
def health():
    return {"direct": "ok", "providers": session.get_providers()}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.30):
    data = np.frombuffer(await file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not a decodable image")
    h, w = img.shape[:2]

    x, r, top, left = preprocess(img)
    t0 = time.perf_counter()
    raw = session.run(None, {"images": x})[0][0]
    infer_ms = (time.perf_counter() - t0) * 1000

    dets = decode(raw, r, top, left, w, h, conf_thr=conf)
    return {"detections": dets, "count": len(dets),
            "infer_ms": round(infer_ms, 2), "backend": "onnxruntime-direct"}
