# SDXL Worker
Stack: Python, Docker, RunPod Serverless
Status: active
Author: Leila

## Paths
- Entry: handler.py
- Config: Dockerfile, requirements.txt
- GitHub: agntc-labs/sdxl-worker

## Deploy
- Push to main → GitHub Actions builds Docker image → ghcr.io/agntc-labs/sdxl-worker:{sha}
- RunPod template ID: wawoa8so8f, endpoint: mddnakg5nggjmi
- Reset workers: scale to 0 then back to max 3

## Current State
RunPod serverless SDXL image generation. PyTorch/MPS locally, GPU on RunPod. NEVER try local docker push — use GitHub Actions.
