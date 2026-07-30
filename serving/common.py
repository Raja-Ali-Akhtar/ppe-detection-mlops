"""Shared serving code: preprocessing, decode, NMS — numpy only, no torch.

Containers stay light: the gateway and direct server import from here.
Letterbox parameters MUST match training/calibration (letterbox 640, pad 114,
BGR->RGB, /255) — same contract as scripts/build_engines.py.
"""

import numpy as np

CLASSES = ["Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest", "Person", "Mask", "NO-Mask"]
IMGSZ = 640


def letterbox(img, size=IMGSZ):
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, top, left


def preprocess(img_bgr):
    """BGR image -> (1,3,640,640) float32 tensor + letterbox geometry."""
    canvas, r, top, left = letterbox(img_bgr)
    x = canvas[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), r, top, left


def nms(boxes, scores, iou_thr):
    """Pure-numpy NMS. boxes: (N,4) xyxy."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def decode(raw, r, top, left, orig_w, orig_h, conf_thr=0.30, iou_thr=0.45):
    """(11,8400) head output -> list of detection dicts in original-image coords."""
    pred = raw.T                               # (8400, 11)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(1)
    scores = scores_all.max(1)
    m = scores > conf_thr
    if not m.any():
        return []
    box, scores, cls = pred[m, :4], scores[m], cls[m]
    xy1 = box[:, :2] - box[:, 2:] / 2
    xy2 = box[:, :2] + box[:, 2:] / 2
    boxes = np.concatenate([xy1, xy2], 1)

    dets = []
    for c in np.unique(cls):                   # class-aware NMS
        idx = np.where(cls == c)[0]
        for k in nms(boxes[idx], scores[idx], iou_thr):
            i = idx[k]
            x1 = float(np.clip((boxes[i, 0] - left) / r, 0, orig_w))
            y1 = float(np.clip((boxes[i, 1] - top) / r, 0, orig_h))
            x2 = float(np.clip((boxes[i, 2] - left) / r, 0, orig_w))
            y2 = float(np.clip((boxes[i, 3] - top) / r, 0, orig_h))
            dets.append({
                "label": CLASSES[int(c)],
                "confidence": round(float(scores[i]), 4),
                "box_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })
    dets.sort(key=lambda d: -d["confidence"])
    return dets
