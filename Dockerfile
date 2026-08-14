# Cadebot API — src/cadebot (STT + LLM + RAG client)
# Chạy CPU (không có GPU trên máy host, xem docs/performance.md).
# Model weights KHÔNG bake vào image — mount qua volume (xem docker-compose.yml).
FROM python:3.12-slim

# ffmpeg: convert audio (.m4a -> .wav) trong /stt
# libsndfile1: dependency của soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Torch CPU-only wheel — bản mặc định trên PyPI kéo theo CUDA libs (~3GB thừa,
# vô dụng vì host không có GPU).
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["python3", "-m", "cadebot"]
