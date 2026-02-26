# ── RunPod Serverless SDXL Worker ──
# Uses RunPod's official base image (supports all RunPod GPU types).
# Models downloaded from HuggingFace at first cold start.
# ──────────────────────────────────────────────────────────────────────

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# ── OCI labels — link to repo for public visibility ─────────────────
LABEL org.opencontainers.image.source="https://github.com/agntc-labs/sdxl-worker"
LABEL org.opencontainers.image.description="RunPod Serverless SDXL Worker"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# ── Python deps ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy handler ─────────────────────────────────────────────────────
COPY handler.py .

# ── Entry point ──────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
