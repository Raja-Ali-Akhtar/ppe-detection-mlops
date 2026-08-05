# needs CUDA + cuDNN runtime for onnxruntime-gpu — hence the NVIDIA base image
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip3 install --no-cache-dir --retries 10 --timeout 60 fastapi uvicorn[standard] python-multipart \
    numpy opencv-python-headless onnxruntime-gpu

COPY common.py /app/common.py
COPY direct/app.py /app/direct/app.py

EXPOSE 9001
CMD ["uvicorn", "direct.app:app", "--host", "0.0.0.0", "--port", "9001"]
