# ── RunPod Serverless SDXL Worker (slim — models from Network Volume) ──
# Models are loaded from /runpod-volume/models/ at runtime.
# The Docker image is just code + Python deps (~5GB, fast builds).
#
# Build:  docker buildx build --platform linux/amd64 -t sdxl-worker .
# Push:   docker tag sdxl-worker <dockerhub-user>/sdxl-worker:latest
#         docker push <dockerhub-user>/sdxl-worker:latest
# ──────────────────────────────────────────────────────────────────────

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# ── System deps ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy handler ─────────────────────────────────────────────────────
COPY handler.py .

# ── Entry point ──────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
