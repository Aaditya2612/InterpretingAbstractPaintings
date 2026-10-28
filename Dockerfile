# ---- Base image ----
FROM python:3.12-slim

# OpenCV needs these native libraries at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies ----
# CPU-only torch to keep the image lean; use the CUDA index if you have a GPU.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- App ----
COPY app.py prefetch.py ./
COPY core/ ./core/
COPY static/ ./static/

# Pre-download model weights into the image so the first request is fast.
# Set PREFETCH_MODELS=0 to build a smaller image and download on first boot.
ENV HF_CACHE_DIR=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PYTHONUNBUFFERED=1

ARG PREFETCH_MODELS=1
RUN if [ "$PREFETCH_MODELS" != "0" ]; then python prefetch.py; fi

EXPOSE 8000

# Honour $PORT (set by Render and other PaaS); default to 8000 locally.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
