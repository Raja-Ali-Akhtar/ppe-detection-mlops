"""Preprocessing / postprocessing contract tests.

These run on any machine with no GPU, no dataset and no model — they test the
pure functions that every consumer shares (calibration, benchmark, gateway,
direct server). If letterboxing drifts between training and serving, mAP dies
silently; this is the guard against that.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving"))
from common import CLASSES, IMGSZ, decode, letterbox, nms, preprocess  # noqa: E402


def make_image(h, w):
    return np.full((h, w, 3), 200, dtype=np.uint8)


@pytest.mark.parametrize("h,w", [(100, 200), (720, 1280), (640, 640), (1000, 300)])
def test_letterbox_preserves_aspect_and_pads(h, w):
    canvas, r, top, left = letterbox(make_image(h, w), IMGSZ)

    assert canvas.shape == (IMGSZ, IMGSZ, 3), "letterbox must produce a square canvas"
    assert r == pytest.approx(min(IMGSZ / h, IMGSZ / w)), "scale must be the limiting ratio"
    # the resized content must fit inside, centred
    assert 0 <= top <= IMGSZ and 0 <= left <= IMGSZ
    assert round(h * r) + 2 * top <= IMGSZ + 1
    assert round(w * r) + 2 * left <= IMGSZ + 1


def test_letterbox_pad_value_is_114():
    """114-grey is the Ultralytics convention — calibration, training and serving
    must agree or INT8 calibration sees a different distribution than inference."""
    canvas, _, top, left = letterbox(make_image(100, 400), IMGSZ)
    if top > 0:
        assert (canvas[0, :, :] == 114).all()
    if left > 0:
        assert (canvas[:, 0, :] == 114).all()


def test_preprocess_tensor_contract():
    x, r, top, left = preprocess(make_image(480, 640))
    assert x.shape == (1, 3, IMGSZ, IMGSZ), "Triton config declares [3,640,640] + batch"
    assert x.dtype == np.float32, "TYPE_FP32 in config.pbtxt"
    assert 0.0 <= x.min() and x.max() <= 1.0, "must be /255 normalised"


def test_nms_keeps_best_of_overlapping_pair():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [500, 500, 600, 600]], float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, 0.45)
    assert 0 in keep and 2 in keep, "distinct box must survive, best of the pair must survive"
    assert 1 not in keep, "the overlapping weaker box must be suppressed"


def test_nms_keeps_both_when_not_overlapping():
    boxes = np.array([[0, 0, 50, 50], [300, 300, 350, 350]], float)
    keep = nms(boxes, np.array([0.9, 0.8]), 0.45)
    assert len(keep) == 2


def test_decode_maps_boxes_back_to_original_image():
    """A detection at the centre of the letterboxed canvas must come back at the
    centre of the ORIGINAL image — this is the coordinate bug that silently
    shifts every box if the padding maths is wrong."""
    orig_w, orig_h = 1280, 720
    _, r, top, left = letterbox(make_image(orig_h, orig_w), IMGSZ)

    raw = np.zeros((4 + len(CLASSES), 8400), dtype=np.float32)
    cx, cy = IMGSZ / 2, IMGSZ / 2
    raw[:4, 0] = [cx, cy, 100, 80]
    raw[4 + CLASSES.index("NO-Hardhat"), 0] = 0.95

    dets = decode(raw, r, top, left, orig_w, orig_h, conf_thr=0.30)
    assert len(dets) == 1
    d = dets[0]
    assert d["label"] == "NO-Hardhat"
    assert d["confidence"] == pytest.approx(0.95, abs=1e-3)

    x1, y1, x2, y2 = d["box_xyxy"]
    assert (x1 + x2) / 2 == pytest.approx(orig_w / 2, abs=2)
    assert (y1 + y2) / 2 == pytest.approx(orig_h / 2, abs=2)


def test_decode_respects_confidence_threshold():
    raw = np.zeros((4 + len(CLASSES), 8400), dtype=np.float32)
    raw[:4, 0] = [320, 320, 50, 50]
    raw[4, 0] = 0.20
    assert decode(raw, 1.0, 0, 0, 640, 640, conf_thr=0.30) == []


def test_decode_clips_boxes_to_image():
    raw = np.zeros((4 + len(CLASSES), 8400), dtype=np.float32)
    raw[:4, 0] = [10, 10, 400, 400]          # box hangs off the top-left
    raw[4, 0] = 0.9
    d = decode(raw, 1.0, 0, 0, 640, 640)[0]
    x1, y1, x2, y2 = d["box_xyxy"]
    assert x1 >= 0 and y1 >= 0 and x2 <= 640 and y2 <= 640
