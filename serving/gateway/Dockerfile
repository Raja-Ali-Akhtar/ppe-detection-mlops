FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir --retries 10 --timeout 60 fastapi uvicorn[standard] python-multipart \
    numpy opencv-python-headless "tritonclient[grpc]"

COPY common.py /app/common.py
COPY gateway/app.py /app/gateway/app.py

EXPOSE 9000
CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "9000"]
