"""RunPod Serverless SDXL Worker — CyberRealistic Pony v8 pipeline on CUDA.

Replicates the local sdxl_server.py pipeline (MPS) for cloud A100/A40 GPUs:
  - CyberRealistic Pony v8 SDXL + 5 LoRAs + FreeU
  - Clip skip 2, EulerAncestral scheduler, fp16-fix VAE
  - BREAK keyword support for character separation
  - ADetailer face fix (YOLO detect -> img2img crop -> Poisson blend)
  - Returns base64-encoded PNG image

Input schema (matches tools.py call signature):
  {
    "input": {
      "action": "generate" | "adetail",

      # -- generate params --
      "prompt": str,
      "negative": str | null,
      "steps": int (default 35),
      "cfg": float (default 6.0),
      "width": int (default 832),
      "height": int (default 1216),
      "seed": int | null,

      # -- adetail params --
      "image_b64": str,                     # base64-encoded source image
      "face_prompt": str | null,            # text prompt for face refinement
      "denoise": float (default 0.25),
      "steps": int (default 20),
      "prompt": str | null,                 # face prompt override
      "padding": float (default 2.5),
      "conf": float (default 0.40),
    }
  }

Output:
  {
    "image_b64": str,           # base64-encoded PNG
    "model": "cyberrealistic-pony-v8",
    "seed": int,
    "gen_time": float,
    ...
  }
"""

import os
import io
import sys
import time
import json
import base64
import random
import logging
import threading

import runpod

# ── Constants ────────────────────────────────────────────────────────
# Network Volume path — persists across worker restarts
_VOL = "/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/models"
MODEL_PATH = f"{_VOL}/cyberrealistic-pony-v8-sdxl"
LORA_DIR = f"{_VOL}/loras/sdxl"
VAE_PATH = f"{_VOL}/sdxl-vae-fp16-fix"
FACE_MODEL_PATH = f"{_VOL}/yolo/face_yolov8n.pt"

# HuggingFace repos for auto-download
_HF_TOKEN = os.environ.get("HF_TOKEN", "")
_HF_MODEL_REPO = "pure-justin/cyberrealistic-pony-v8-sdxl"
_HF_LORA_REPO = "pure-justin/sdxl-loras"


def _ensure_models():
    """Download models from HuggingFace if not present on Network Volume.

    First cold start: ~3-5 min (downloads ~10GB at datacenter speeds).
    Subsequent starts: instant (cached on Network Volume).
    """
    from huggingface_hub import snapshot_download
    import subprocess

    os.makedirs(_VOL, exist_ok=True)

    # 1. CyberRealistic Pony v8 (~6.5GB)
    if not os.path.exists(os.path.join(MODEL_PATH, "model_index.json")):
        log.info("Downloading CyberRealistic Pony v8 from HuggingFace...")
        snapshot_download(
            _HF_MODEL_REPO, local_dir=MODEL_PATH,
            local_dir_use_symlinks=False, token=_HF_TOKEN or None,
        )
        log.info("Model downloaded to %s", MODEL_PATH)
    else:
        log.info("Model found at %s", MODEL_PATH)

    # 2. LoRAs (~950MB)
    if not os.path.exists(os.path.join(LORA_DIR, "good_hands_pony.safetensors")):
        log.info("Downloading LoRAs from HuggingFace...")
        os.makedirs(LORA_DIR, exist_ok=True)
        snapshot_download(
            _HF_LORA_REPO, local_dir=LORA_DIR,
            local_dir_use_symlinks=False, token=_HF_TOKEN or None,
        )
        log.info("LoRAs downloaded to %s", LORA_DIR)
    else:
        log.info("LoRAs found at %s", LORA_DIR)

    # 3. VAE (public, ~335MB)
    if not os.path.exists(os.path.join(VAE_PATH, "diffusion_pytorch_model.safetensors")):
        log.info("Downloading sdxl-vae-fp16-fix...")
        snapshot_download(
            "madebyollin/sdxl-vae-fp16-fix", local_dir=VAE_PATH,
            local_dir_use_symlinks=False,
        )
        log.info("VAE downloaded")
    else:
        log.info("VAE found at %s", VAE_PATH)

    # 4. YOLO face detection (~6MB) — optional, ADetailer won't work without it
    if not os.path.exists(FACE_MODEL_PATH):
        log.info("Downloading YOLOv8n face model...")
        os.makedirs(os.path.dirname(FACE_MODEL_PATH), exist_ok=True)
        try:
            subprocess.run([
                "wget", "-q", "--timeout=15", "-O", FACE_MODEL_PATH,
                "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
            ], check=True)
            log.info("YOLO downloaded")
        except Exception as e:
            log.warning("YOLO download failed (%s) — ADetailer disabled", e)
            if os.path.exists(FACE_MODEL_PATH):
                os.remove(FACE_MODEL_PATH)  # Remove partial download
    else:
        log.info("YOLO found at %s", FACE_MODEL_PATH)

LORA_CONFIG = {
    "good_hands":  ("good_hands_pony.safetensors",         0.9),
    "realism":     ("zy_realism_enhancer_v2.safetensors",  0.85),
    "vivid":       ("vivid_realism_color.safetensors",     0.65),
    "detail":      ("detail_slider_pony.safetensors",      0.7),
    "pony_detail": ("pony_detail_v2.safetensors",          0.5),
}

# Quality prefix — 17 CLIP tokens. Verified with CLIPTokenizer.
# score_9 + score_8_up pull quality up (score_7_up redundant).
# source_pony = Pony model quality trigger. photorealistic = realism anchor.
# No emphasis weights — saves ~4 tokens vs (photorealistic:1.4).
PONY_QUALITY_PREFIX = (
    "score_9, score_8_up, source_pony, photorealistic, "
)

# CRITICAL: CLIP hard limit = 75 tokens. Tags beyond 75 are INVISIBLE.
# 66 CLIP tokens — verified with CLIPTokenizer. Leaves ~5 tokens for per-agent negative.
# Score penalties REMOVED (redundant — positive has score_9 + score_8_up pulling quality up).
# Emphasis weights REMOVED (saves ~10 tokens vs weighted versions).
# Anti-style FIRST (most important), then cosmetic, then anatomy, then artifacts.
DEFAULT_NEGATIVE = (
    "source_cartoon, source_anime, source_furry, "
    "anime, cartoon, 3d render, illustration, "
    "painting, drawing, digital art, CGI, "
    "tan lines, bikini lines, "
    "bad anatomy, bad hands, deformed, ugly, blurry, "
    "bad eyes, extra fingers, fused fingers, "
    "text, watermark, plastic skin"
)

MULTI_CHAR_NEGATIVE = (
    "merged bodies, fused bodies, conjoined, conjoined twins, "
    "extra body, multiple heads, two heads on one body, shared body, "
    "fused faces, merged faces, split face, half face, "
    "wrong number of limbs, body horror, anatomically impossible, "
    "extra arms, extra legs, three arms, three legs, "
    "clone, duplicate person, same face on both characters"
)

FREEU_S1, FREEU_S2, FREEU_B1, FREEU_B2 = 0.6, 0.4, 1.1, 1.2

log = logging.getLogger("sdxl-worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

# ── Global pipeline state ────────────────────────────────────────────
_pipe = None
_img2img_pipe = None
_face_model = None
_pipe_lock = threading.Lock()
_gen_count = 0
_torch = None
_Image = None


# ── Helpers ──────────────────────────────────────────────────────────

def _b64_to_pil(b64_str):
    """Decode a base64 string to a PIL Image (RGB)."""
    data = base64.b64decode(b64_str)
    return _Image.open(io.BytesIO(data)).convert("RGB")


def _pil_to_b64(img):
    """Encode a PIL Image to a base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Pipeline Loading ─────────────────────────────────────────────────

def _load_pipeline():
    """Load the full SDXL pipeline with all enhancements. Called once at container startup."""
    global _pipe, _img2img_pipe, _face_model, _torch, _Image

    import torch
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        EulerAncestralDiscreteScheduler,
        AutoencoderKL,
    )
    from PIL import Image

    _torch = torch
    _Image = Image

    t0 = time.time()
    log.info("Loading CyberRealistic Pony v8 SDXL pipeline (CUDA)...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True,
    )

    # Scheduler — Euler Ancestral (matches CyberRealistic Pony v8 community standard)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    log.info("Scheduler: Euler Ancestral")

    # VAE — fp16-fix for sharper colors and fewer skin tone artifacts
    if os.path.exists(VAE_PATH):
        pipe.vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=torch.float16)
        log.info("VAE: sdxl-vae-fp16-fix")

    # Clip skip 2
    pipe.text_encoder.config.num_hidden_layers -= 1
    log.info("Clip skip 2 enabled")

    # Move to CUDA
    pipe = pipe.to("cuda")

    # FreeU
    pipe.enable_freeu(s1=FREEU_S1, s2=FREEU_S2, b1=FREEU_B1, b2=FREEU_B2)
    log.info(
        "FreeU enabled (s1=%.1f, s2=%.1f, b1=%.1f, b2=%.1f)",
        FREEU_S1, FREEU_S2, FREEU_B1, FREEU_B2,
    )

    # LoRAs — load each by weight_name
    loaded = []
    for name, (fname, weight) in LORA_CONFIG.items():
        fpath = os.path.join(LORA_DIR, fname)
        if os.path.exists(fpath):
            pipe.load_lora_weights(LORA_DIR, weight_name=fname, adapter_name=name)
            loaded.append(name)
            log.info("LoRA: %s (%.2f)", name, weight)
    if loaded:
        adapter_weights = [LORA_CONFIG[n][1] for n in loaded]
        pipe.set_adapters(loaded, adapter_weights=adapter_weights)

    # Img2Img pipeline (shares components with txt2img, used by ADetailer)
    _img2img_pipe = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )
    _img2img_pipe = _img2img_pipe.to("cuda")
    log.info("Img2Img pipeline ready (shared components, for ADetailer)")

    # Face detection model (YOLOv8 face)
    if os.path.exists(FACE_MODEL_PATH):
        from ultralytics import YOLO

        _face_model = YOLO(FACE_MODEL_PATH)
        log.info("Face detection model loaded: %s", FACE_MODEL_PATH)

    t_load = time.time() - t0
    _pipe = pipe
    log.info(
        "Pipeline ready in %.1fs (LoRAs=%d, FreeU=True, ADetailer=%s)",
        t_load, len(loaded), _face_model is not None,
    )


# ── Generation ───────────────────────────────────────────────────────

def _generate(params):
    """Generate an image. Returns dict with image_b64 and metadata."""
    global _gen_count

    prompt = params.get("prompt", "")
    steps = params.get("steps", 35)
    width = params.get("width", 832)
    height = params.get("height", 1216)
    cfg = params.get("cfg", 6.0)
    seed = params.get("seed")
    negative = params.get("negative")

    if not prompt:
        return {"error": "Empty prompt"}

    # Build full prompt — quality prefix prepended
    full_prompt = PONY_QUALITY_PREFIX + prompt
    # Append custom negative to default (don't replace — default has critical tags)
    neg = DEFAULT_NEGATIVE + (", " + negative if negative else "")

    # ── Token count + hard trim to 75 tokens (CLIP limit) ──────────
    try:
        _tok = _pipe.tokenizer
        _ids = _tok(full_prompt, truncation=False, add_special_tokens=False)["input_ids"]
        _exact_tokens = len(_ids)
        if _exact_tokens > 75:
            _trimmed_ids = _ids[:75]
            full_prompt = _tok.decode(_trimmed_ids, skip_special_tokens=True)
            log.warning("CLIP tokens: %d/75 — trimmed last %d tokens", _exact_tokens, _exact_tokens - 75)
        else:
            log.info("CLIP tokens: %d/75", _exact_tokens)
    except Exception:
        pass

    # Seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    with _pipe_lock:
        _gen_count += 1
        gen_id = _gen_count

        generator = _torch.Generator("cuda").manual_seed(seed)

        log.info(
            "[gen #%d] %dx%d, %d steps, cfg=%.1f, seed=%d -- %s",
            gen_id, width, height, steps, cfg, seed, prompt[:100],
        )

        t0 = time.time()

        gen_kwargs = dict(
            prompt=full_prompt,
            negative_prompt=neg,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            generator=generator,
        )

        try:
            image = _pipe(**gen_kwargs).images[0]
        except Exception as gen_err:
            t_gen = time.time() - t0
            log.error("[gen #%d] Failed in %.1fs: %s", gen_id, t_gen, gen_err)
            return {"error": "Generation failed: %s" % str(gen_err)}

        t_gen = time.time() - t0

    # Encode to base64
    image_b64 = _pil_to_b64(image)

    log.info("[gen #%d] Done in %.1fs (%d bytes b64)", gen_id, t_gen, len(image_b64))

    return {
        "image_b64": image_b64,
        "model": "cyberrealistic-pony-v8",
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "seed": seed,
        "gen_time": round(t_gen, 1),
        "freeu": True,
    }


# ── ADetailer ────────────────────────────────────────────────────────

def _adetail(params):
    """ADetailer: detect faces, crop each, img2img with text prompt, Poisson-blend back.

    Uses OpenCV seamlessClone for invisible boundaries. Crop-based img2img with
    text-only face prompts (no IP-Adapter). Low denoise preserves skin/lighting.

    Params (in input dict):
        image_b64: base64-encoded source image
        face_prompt: text prompt for face refinement (from character looks YAML)
        denoise: denoising strength (default 0.25)
        steps: img2img steps (default 20)
        prompt: optional face prompt override (alias for face_prompt)
        padding: bbox expansion multiplier (default 2.5)
        conf: YOLO confidence threshold (default 0.40)
    """
    import cv2
    import numpy as np

    image_b64 = params.get("image_b64", "")
    denoise = params.get("denoise", 0.25)
    steps = params.get("steps", 20)
    face_prompt = params.get("face_prompt") or params.get("prompt")
    padding_mult = params.get("padding", 2.5)
    conf_thresh = params.get("conf", 0.40)

    if not image_b64:
        return {"error": "No image_b64 provided"}
    if not _face_model:
        return {"error": "Face detection model not loaded"}
    if not _img2img_pipe:
        return {"error": "Img2Img pipeline not ready"}

    # Decode source image
    src_image = _b64_to_pil(image_b64)
    w, h = src_image.size

    # Detect faces
    results = _face_model(src_image, conf=conf_thresh, verbose=False)
    faces = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            faces.append((x1, y1, x2, y2, conf))

    if not faces:
        log.info("[adetail] No faces detected")
        return {
            "image_b64": _pil_to_b64(src_image),
            "faces_fixed": 0,
            "faces_detected": 0,
        }

    # Sort left-to-right
    faces.sort(key=lambda f: (f[0] + f[2]) / 2)

    if not face_prompt:
        face_prompt = (
            PONY_QUALITY_PREFIX
            + "1person, detailed face, beautiful eyes, sharp detailed eyes, "
            "symmetric eyes, realistic eyes, (detailed skin:1.2), "
            "photorealistic face, sharp focus"
        )

    face_neg = (
        "(bad eyes:1.4), asymmetric eyes, uneven eyes, misaligned eyes, cross-eyed, "
        "wonky eyes, glowing eyes, dead eyes, empty eyes, extra eyes, "
        "deformed face, ugly face, blurry face, poorly drawn face, "
        "bad anatomy, mutation, disfigured, extra fingers, missing fingers"
    )

    num_to_fix = len(faces)
    log.info("[adetail] Detected %d faces, will fix all with text prompt", num_to_fix)

    fixed_count = 0
    errors = []
    current_cv = cv2.cvtColor(np.array(src_image), cv2.COLOR_RGB2BGR)

    with _pipe_lock:
        _img2img_pipe.to(dtype=_torch.float32)

        for i in range(num_to_fix):
            x1, y1, x2, y2, conf = faces[i]

            face_w = x2 - x1
            face_h = y2 - y1

            pad_x = face_w * (padding_mult - 1) / 2
            pad_y = face_h * (padding_mult - 1) / 2
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y * 0.8))
            cx2 = min(w, int(x2 + pad_x))
            cy2 = min(h, int(y2 + pad_y * 1.2))

            crop_w = cx2 - cx1
            crop_h = cy2 - cy1

            scale = max(512 / min(crop_w, crop_h), 1.0)
            target_w = ((int(crop_w * scale) + 7) // 8) * 8
            target_h = ((int(crop_h * scale) + 7) // 8) * 8
            target_w = min(768, max(512, target_w))
            target_h = min(768, max(512, target_h))

            current_pil = _Image.fromarray(cv2.cvtColor(current_cv, cv2.COLOR_BGR2RGB))
            face_crop = current_pil.crop((cx1, cy1, cx2, cy2))
            face_crop_resized = face_crop.resize((target_w, target_h), _Image.LANCZOS)

            gen = _torch.Generator("cuda").manual_seed(42 + i)

            log.info(
                "[adetail] Fixing face %d/%d: crop=(%d,%d,%d,%d) %dx%d -> %dx%d denoise=%.2f",
                i + 1, num_to_fix, cx1, cy1, cx2, cy2,
                crop_w, crop_h, target_w, target_h, denoise,
            )

            try:
                fixed_crop = _img2img_pipe(
                    prompt=face_prompt,
                    negative_prompt=face_neg,
                    image=face_crop_resized,
                    num_inference_steps=steps,
                    guidance_scale=6.0,
                    strength=denoise,
                    generator=gen,
                ).images[0]

                fixed_crop = fixed_crop.resize((crop_w, crop_h), _Image.LANCZOS)
                fixed_cv = cv2.cvtColor(np.array(fixed_crop), cv2.COLOR_RGB2BGR)

                mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                margin_x = int(crop_w * 0.10)
                margin_y = int(crop_h * 0.10)
                center = (crop_w // 2, crop_h // 2)
                axes = ((crop_w // 2) - margin_x, (crop_h // 2) - margin_y)
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3, crop_w * 0.03))
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                dest_center = (cx1 + crop_w // 2, cy1 + crop_h // 2)
                current_cv = cv2.seamlessClone(
                    fixed_cv, current_cv, mask, dest_center, cv2.NORMAL_CLONE,
                )

                fixed_count += 1
                log.info("[adetail] Face %d Poisson-blended successfully", i + 1)

            except Exception as e:
                log.error("[adetail] Face %d failed: %s", i, e, exc_info=True)
                errors.append(str(e))
                continue

        _img2img_pipe.to(dtype=_torch.float16)

    result_image = _Image.fromarray(cv2.cvtColor(current_cv, cv2.COLOR_BGR2RGB))
    image_b64 = _pil_to_b64(result_image)

    log.info("[adetail] Fixed %d/%d faces", fixed_count, len(faces))

    return {
        "image_b64": image_b64,
        "faces_detected": len(faces),
        "faces_fixed": fixed_count,
        "denoise": denoise,
        "steps": steps,
        "errors": errors if errors else None,
    }


# ── RunPod Handler ───────────────────────────────────────────────────

def handler(job):
    """RunPod serverless handler. Dispatches to generate or adetail."""
    job_input = job.get("input", {})
    action = job_input.get("action", "generate")

    if _pipe is None:
        return {"error": "Pipeline still loading, please retry"}

    try:
        if action == "adetail":
            return _adetail(job_input)
        else:
            return _generate(job_input)
    except Exception as e:
        log.error("Handler error: %s", e, exc_info=True)
        return {"error": str(e)}


# ── Startup ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Initializing SDXL worker (CUDA)...")
    log.info("Model volume: %s", _VOL)
    _ensure_models()  # Download from HuggingFace if not cached on volume
    _load_pipeline()
    log.info("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
