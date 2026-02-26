# ── RunPod Serverless SDXL Worker (slim — models from Network Volume) ──
# Models are loaded from /runpod-volume/models/ at runtime.
# Uses runtime CUDA image (not devel) — ~4GB base, inference only.
# ──────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# ── OCI labels — link to repo for public visibility ─────────────────
LABEL org.opencontainers.image.source="https://github.com/agntc-labs/sdxl-worker"
LABEL org.opencontainers.image.description="RunPod Serverless SDXL Worker"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# ── System deps ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy handler ─────────────────────────────────────────────────────
COPY handler.py .

# ── Entry point ──────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
