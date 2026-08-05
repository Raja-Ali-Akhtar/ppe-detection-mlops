"""FastAPI gateway: business API in front of Triton.

POST /detect (JPEG/PNG) -> JSON detections. The gateway owns pre/post-processing
and NOTHING else — inference belongs to Triton (dynamic batching, instances,
metrics). If this file grows a brain, you're rebuilding Triton in Python.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tritonclient.grpc as triton_grpc
from fastapi import FastAPI, File, HTTPException, UploadFile

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import decode, preprocess

# gRPC, not HTTP: tritonclient.http is gevent-based and collides with FastAPI's
# asyncio loop ("Cannot switch to a different thread"). gRPC client has no
# gevent, is faster on binary tensors, and works from sync endpoints (FastAPI
# runs plain `def` handlers in its threadpool).
TRITON_URL = os.environ.get("TRITON_URL", "localhost:8001")
MODEL = "ppe_detector"

app = FastAPI(title="PPE Detection Gateway")
client = triton_grpc.InferenceServerClient(url=TRITON_URL)


@app.get("/health")
def health():
    try:
        ready = client.is_model_ready(MODEL)
    except Exception:
        ready = False
    return {"gateway": "ok", "triton_model_ready": ready}


@app.post("/detect")
def detect(file: UploadFile = File(...), conf: float = 0.30):
    data = np.frombuffer(file.file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not a decodable image")
    h, w = img.shape[:2]

    x, r, top, left = preprocess(img)
    inp = triton_grpc.InferInput("images", x.shape, "FP32")
    inp.set_data_from_numpy(x)
    t0 = time.perf_counter()
    result = client.infer(MODEL, inputs=[inp])
    infer_ms = (time.perf_counter() - t0) * 1000

    raw = result.as_numpy("output0")[0]
    dets = decode(raw, r, top, left, w, h, conf_thr=conf)
    return {"detections": dets, "count": len(dets),
            "infer_ms": round(infer_ms, 2), "backend": "triton"}
