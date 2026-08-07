"""Concurrent load test: Triton-gateway vs FastAPI-direct, same image, same GPU.

Usage:
    python scripts/load_test.py --workers 16 --requests 200

Simulates N independent clients (thread pool) POSTing the same image. Reports
throughput and latency percentiles per backend. Concurrency is the whole
point: batch-1-per-request architectures look fine alone and fall behind when
requests overlap — or so the theory says. This script checks.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
IMAGE = next((ROOT / "data/processed/ppe-7cls-v1/valid/images").glob("youtube-455*.jpg"))

TARGETS = {
    "triton-gateway": "http://" + (os.environ.get("TARGET_HOST","localhost")) + ":9000/detect",
    "fastapi-direct": "http://" + (os.environ.get("TARGET_HOST","localhost")) + ":9001/detect",
}


def one_request(url, payload):
    t0 = time.perf_counter()
    r = requests.post(url, files={"file": ("img.jpg", payload, "image/jpeg")}, timeout=60)
    ms = (time.perf_counter() - t0) * 1000
    return ms, r.status_code


def bench(url, payload, workers, total):
    # warmup: fill kernels/caches before measuring
    for _ in range(8):
        one_request(url, payload)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _: one_request(url, payload), range(total)))
    wall = time.perf_counter() - t0

    lat = np.array([ms for ms, _ in results])
    errors = sum(1 for _, code in results if code != 200)
    return {
        "throughput_rps": total / wall,
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--requests", type=int, default=200)
    args = parser.parse_args()

    payload = IMAGE.read_bytes()
    print(f"image: {IMAGE.name} ({len(payload)/1024:.0f} KB), "
          f"{args.workers} concurrent clients, {args.requests} requests each backend\n")

    for name, url in TARGETS.items():
        try:
            r = bench(url, payload, args.workers, args.requests)
        except Exception as e:
            print(f"{name}: FAILED - {e}")
            continue
        print(f"{name:16s}  {r['throughput_rps']:6.1f} req/s   "
              f"p50 {r['p50_ms']:7.1f} ms   p95 {r['p95_ms']:7.1f} ms   "
              f"p99 {r['p99_ms']:7.1f} ms   errors {r['errors']}")


if __name__ == "__main__":
    main()
