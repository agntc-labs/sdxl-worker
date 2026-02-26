#!/bin/bash
# ── Populate RunPod Network Volume with SDXL models ──────────────────
# Run this ONCE on a temporary GPU Pod with the Network Volume mounted.
# The volume will be at /runpod-volume/. After this, serverless workers
# load models from the volume with zero download time.
#
# Usage:
#   1. Create a Network Volume on RunPod (20GB, same region as endpoint)
#   2. Start a GPU Pod (cheapest: A4000, ~$0.20/hr) with the volume
#   3. SSH in and run: bash setup_volume.sh
#   4. Stop the pod (volume persists)
# ─────────────────────────────────────────────────────────────────────

set -e

VOL="/runpod-volume/models"
echo "=== Setting up models in $VOL ==="

mkdir -p "$VOL/cyberrealistic-pony-v8-sdxl" \
         "$VOL/loras/sdxl" \
         "$VOL/ip-adapter-sdxl/sdxl_models" \
         "$VOL/ip-adapter-sdxl/models/image_encoder" \
         "$VOL/sdxl-vae-fp16-fix" \
         "$VOL/yolo"

pip install -q huggingface-hub

# ── 1. VAE (public) ──
echo ">>> Downloading sdxl-vae-fp16-fix..."
huggingface-cli download madebyollin/sdxl-vae-fp16-fix \
    --local-dir "$VOL/sdxl-vae-fp16-fix" \
    --local-dir-use-symlinks False
echo "    VAE done."

# ── 2. IP-Adapter (public) ──
echo ">>> Downloading IP-Adapter Plus Face SDXL..."
huggingface-cli download h94/IP-Adapter \
    --include "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors" \
    --local-dir "$VOL/ip-adapter-sdxl" \
    --local-dir-use-symlinks False

echo ">>> Downloading CLIP ViT-H image encoder..."
huggingface-cli download h94/IP-Adapter \
    --include "models/image_encoder/*" \
    --local-dir "$VOL/ip-adapter-sdxl" \
    --local-dir-use-symlinks False
echo "    IP-Adapter done."

# ── 3. YOLO face detection ──
echo ">>> Downloading YOLOv8n face model..."
wget -q -O "$VOL/yolo/face_yolov8n.pt" \
    https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt
echo "    YOLO done."

# ── 4. CyberRealistic Pony v8 (private — upload first) ──
# Models are on private HuggingFace repos (pure-justin account):
#   pure-justin/cyberrealistic-pony-v8-sdxl
#   pure-justin/sdxl-loras
#
if [ -n "$HF_TOKEN" ]; then
    echo ">>> Downloading CyberRealistic Pony v8 from HuggingFace..."
    huggingface-cli download pure-justin/cyberrealistic-pony-v8-sdxl \
        --local-dir "$VOL/cyberrealistic-pony-v8-sdxl" \
        --local-dir-use-symlinks False \
        --token "$HF_TOKEN"
    echo "    Model done."

    echo ">>> Downloading LoRAs from HuggingFace..."
    huggingface-cli download pure-justin/sdxl-loras \
        --local-dir "$VOL/loras/sdxl" \
        --local-dir-use-symlinks False \
        --token "$HF_TOKEN"
    echo "    LoRAs done."
else
    echo "!!! HF_TOKEN not set — skipping model + LoRA download."
    echo "    Run with: HF_TOKEN=hf_xxx bash setup_volume.sh"
fi

# ── Verify ──
echo ""
echo "=== Volume contents ==="
du -sh "$VOL"/*
echo ""
echo "=== Model check ==="
[ -f "$VOL/cyberrealistic-pony-v8-sdxl/model_index.json" ] && echo "✓ CyberRealistic Pony v8" || echo "✗ CyberRealistic Pony v8 MISSING"
[ -f "$VOL/loras/sdxl/good_hands_pony.safetensors" ] && echo "✓ LoRAs" || echo "✗ LoRAs MISSING"
[ -f "$VOL/ip-adapter-sdxl/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors" ] && echo "✓ IP-Adapter" || echo "✗ IP-Adapter MISSING"
[ -f "$VOL/sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors" ] && echo "✓ VAE" || echo "✗ VAE MISSING"
[ -f "$VOL/yolo/face_yolov8n.pt" ] && echo "✓ YOLO" || echo "✗ YOLO MISSING"
echo ""
echo "=== Done! You can now stop this pod. ==="
