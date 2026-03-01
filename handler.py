"""RunPod Serverless SDXL Worker — CyberRealistic Pony v8 on CUDA.

Pipeline: CyberRealistic Pony v8 SDXL + 5 LoRAs + FreeU + IP-Adapter Plus Face
Settings: Clip skip 2, EulerAncestral, fp16-fix VAE, CFG 6.0
Features: CLIP hard trim (75 tokens), VRAM management, ADetailer, multi-face masking

Input schema:
  {
    "input": {
      "action": "generate" | "adetail",
      "prompt": str,
      "negative": str | null,
      "steps": int (default 35),
      "cfg": float (default 6.0),
      "width": int (default 832),
      "height": int (default 1216),
      "seed": int | null,
      "face_ref_b64": [str] | str | null,
      "face_scale": float (default 0.5),
      "mask_mode": "horizontal" | "vertical" (default "horizontal"),
    }
  }
"""

import os
import io
import sys
import time
import base64
import random
import logging
import threading

import runpod

# ── Constants ────────────────────────────────────────────────────────
_VOL = "/runpod-volume/models" if os.path.isdir("/runpod-volume") else "/models"
MODEL_PATH = f"{_VOL}/cyberrealistic-pony-v8-sdxl"
LORA_DIR = f"{_VOL}/loras/sdxl"
IP_ADAPTER_DIR = f"{_VOL}/ip-adapter-sdxl"
VAE_PATH = f"{_VOL}/sdxl-vae-fp16-fix"
FACE_MODEL_PATH = f"{_VOL}/yolo/face_yolov8n.pt"

_HF_TOKEN = os.environ.get("HF_TOKEN", "")
_HF_MODEL_REPO = "pure-justin/cyberrealistic-pony-v8-sdxl"
_HF_LORA_REPO = "pure-justin/sdxl-loras"

CLIP_TOKEN_LIMIT = 75

# Quality prefix — 17 CLIP tokens verified.
PONY_QUALITY_PREFIX = "score_9, score_8_up, source_pony, photorealistic, "

# 48 CLIP tokens — verified. Anti-style first, then cosmetic, then anatomy.
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
    "merged bodies, fused bodies, conjoined, extra body, "
    "multiple heads, two heads on one body, shared body, "
    "fused faces, wrong number of limbs, extra arms, extra legs, "
    "clone, duplicate person, same face on both characters"
)

LORA_CONFIG = {
    "good_hands":  ("good_hands_pony.safetensors",         0.9),
    "realism":     ("zy_realism_enhancer_v2.safetensors",  0.85),
    "vivid":       ("vivid_realism_color.safetensors",     0.65),
    "detail":      ("detail_slider_pony.safetensors",      0.7),
    "pony_detail": ("pony_detail_v2.safetensors",          0.5),
}

FREEU_S1, FREEU_S2, FREEU_B1, FREEU_B2 = 0.6, 0.4, 1.1, 1.2
IP_ADAPTER_SCALE = 0.5

log = logging.getLogger("sdxl-worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

# ── Global pipeline state ────────────────────────────────────────────
_pipe = None
_img2img_pipe = None
_face_model = None
_pipe_lock = threading.Lock()
_ip_adapter_loaded = False
_gen_count = 0
_torch = None
_Image = None


# ── Helpers ──────────────────────────────────────────────────────────

def _b64_to_pil(b64_str):
    data = base64.b64decode(b64_str)
    return _Image.open(io.BytesIO(data)).convert("RGB")


def _pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _vram_status():
    """Return human-readable VRAM usage string."""
    try:
        alloc = _torch.cuda.memory_allocated() / 1024**3
        reserved = _torch.cuda.memory_reserved() / 1024**3
        total = _torch.cuda.get_device_properties(0).total_mem / 1024**3
        free = total - reserved
        return "%.1f/%.1fGB used (%.1fGB reserved, %.1fGB free)" % (alloc, total, reserved, free)
    except Exception:
        return "unavailable"


def _clip_trim(text, label, tokenizer):
    """Tokenize, log detail, hard-trim to 75 tokens. Returns (trimmed_text, token_count, clipped_text)."""
    try:
        ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
        count = len(ids)
        if count > CLIP_TOKEN_LIMIT:
            kept_ids = ids[:CLIP_TOKEN_LIMIT]
            clipped_ids = ids[CLIP_TOKEN_LIMIT:]
            trimmed = tokenizer.decode(kept_ids, skip_special_tokens=True)
            clipped = tokenizer.decode(clipped_ids, skip_special_tokens=True)
            log.warning("[CLIP/%s] %d/%d tokens — TRIMMED %d tokens", label, count, CLIP_TOKEN_LIMIT, count - CLIP_TOKEN_LIMIT)
            log.warning("[CLIP/%s] KEPT:    %s", label, trimmed)
            log.warning("[CLIP/%s] CLIPPED: %s", label, clipped)
            return trimmed, count, clipped
        else:
            log.info("[CLIP/%s] %d/%d tokens — OK", label, count, CLIP_TOKEN_LIMIT)
            return text, count, None
    except Exception as e:
        log.warning("[CLIP/%s] Tokenizer error: %s", label, e)
        return text, -1, None


# ── Model Download ───────────────────────────────────────────────────

def _ensure_models():
    """Download models from HuggingFace if not present on Network Volume."""
    from huggingface_hub import snapshot_download
    import subprocess

    os.makedirs(_VOL, exist_ok=True)

    # 1. CyberRealistic Pony v8 (~6.5GB)
    if not os.path.exists(os.path.join(MODEL_PATH, "model_index.json")):
        log.info("[download] CyberRealistic Pony v8...")
        snapshot_download(_HF_MODEL_REPO, local_dir=MODEL_PATH, local_dir_use_symlinks=False, token=_HF_TOKEN or None)
    log.info("[download] Model: %s", MODEL_PATH)

    # 2. LoRAs (~950MB)
    if not os.path.exists(os.path.join(LORA_DIR, "good_hands_pony.safetensors")):
        log.info("[download] LoRAs...")
        os.makedirs(LORA_DIR, exist_ok=True)
        snapshot_download(_HF_LORA_REPO, local_dir=LORA_DIR, local_dir_use_symlinks=False, token=_HF_TOKEN or None)
    log.info("[download] LoRAs: %s", LORA_DIR)

    # 3. VAE (~335MB)
    if not os.path.exists(os.path.join(VAE_PATH, "diffusion_pytorch_model.safetensors")):
        log.info("[download] sdxl-vae-fp16-fix...")
        snapshot_download("madebyollin/sdxl-vae-fp16-fix", local_dir=VAE_PATH, local_dir_use_symlinks=False)
    log.info("[download] VAE: %s", VAE_PATH)

    # 4. IP-Adapter (~3.2GB)
    _ip_file = os.path.join(IP_ADAPTER_DIR, "sdxl_models", "ip-adapter-plus-face_sdxl_vit-h.safetensors")
    if not os.path.exists(_ip_file):
        log.info("[download] IP-Adapter Plus Face...")
        snapshot_download(
            "h94/IP-Adapter",
            allow_patterns=["sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors", "models/image_encoder/*"],
            local_dir=IP_ADAPTER_DIR, local_dir_use_symlinks=False,
        )
    log.info("[download] IP-Adapter: %s", IP_ADAPTER_DIR)

    # 5. YOLO face detection (~6MB)
    if not os.path.exists(FACE_MODEL_PATH):
        log.info("[download] YOLOv8n face model...")
        os.makedirs(os.path.dirname(FACE_MODEL_PATH), exist_ok=True)
        try:
            subprocess.run(["wget", "-q", "--timeout=15", "-O", FACE_MODEL_PATH,
                            "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"], check=True)
        except Exception as e:
            log.warning("[download] YOLO failed (%s) — ADetailer disabled", e)
            if os.path.exists(FACE_MODEL_PATH):
                os.remove(FACE_MODEL_PATH)
    log.info("[download] YOLO: %s (exists=%s)", FACE_MODEL_PATH, os.path.exists(FACE_MODEL_PATH))


# ── Pipeline Loading ─────────────────────────────────────────────────

def _load_pipeline():
    """Load SDXL pipeline with VRAM management for 24GB GPUs."""
    global _pipe, _img2img_pipe, _face_model, _ip_adapter_loaded, _torch, _Image

    # Enable expandable segments BEFORE any CUDA allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    log.info("[load] Starting pipeline load...")
    log.info("[load] VRAM before load: %s", _vram_status())

    # Check IP-Adapter files
    encoder_path = os.path.join(IP_ADAPTER_DIR, "models", "image_encoder")
    adapter_file = os.path.join(IP_ADAPTER_DIR, "sdxl_models", "ip-adapter-plus-face_sdxl_vit-h.safetensors")
    has_ip_adapter = os.path.exists(encoder_path) and os.path.exists(adapter_file)
    log.info("[load] IP-Adapter files present: %s", has_ip_adapter)

    # Load base pipeline
    if has_ip_adapter:
        from transformers import CLIPVisionModelWithProjection
        log.info("[load] Loading CLIP ViT-H image encoder...")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(encoder_path, torch_dtype=torch.float16)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_PATH, image_encoder=image_encoder, torch_dtype=torch.float16, use_safetensors=True)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)

    # Scheduler
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # VAE
    if os.path.exists(VAE_PATH):
        pipe.vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=torch.float16)

    # Clip skip 2
    pipe.text_encoder.config.num_hidden_layers -= 1

    # Move to CUDA
    pipe = pipe.to("cuda")
    log.info("[load] VRAM after model load: %s", _vram_status())

    # VRAM management — critical for 24GB GPUs with IP-Adapter
    pipe.enable_vae_slicing()    # Process VAE in slices (saves ~2GB peak)
    pipe.enable_vae_tiling()     # Tile large images through VAE
    log.info("[load] VAE slicing + tiling enabled")

    # FreeU
    pipe.enable_freeu(s1=FREEU_S1, s2=FREEU_S2, b1=FREEU_B1, b2=FREEU_B2)

    # LoRAs
    loaded = []
    for name, (fname, weight) in LORA_CONFIG.items():
        fpath = os.path.join(LORA_DIR, fname)
        if os.path.exists(fpath):
            pipe.load_lora_weights(LORA_DIR, weight_name=fname, adapter_name=name)
            loaded.append(name)
    if loaded:
        adapter_weights = [LORA_CONFIG[n][1] for n in loaded]
        pipe.set_adapters(loaded, adapter_weights=adapter_weights)
    log.info("[load] LoRAs: %s", ", ".join("%s(%.2f)" % (n, LORA_CONFIG[n][1]) for n in loaded))

    # IP-Adapter
    if has_ip_adapter:
        pipe.load_ip_adapter(IP_ADAPTER_DIR, subfolder="sdxl_models",
                             weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors")
        pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
        _ip_adapter_loaded = True
        log.info("[load] IP-Adapter loaded (scale=%.2f)", IP_ADAPTER_SCALE)

    log.info("[load] VRAM after LoRAs + IP-Adapter: %s", _vram_status())

    # Img2Img pipeline (shared components)
    _img2img_pipe = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae, text_encoder=pipe.text_encoder, text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer, tokenizer_2=pipe.tokenizer_2, unet=pipe.unet,
        scheduler=pipe.scheduler, image_encoder=getattr(pipe, "image_encoder", None),
        feature_extractor=getattr(pipe, "feature_extractor", None),
    )
    _img2img_pipe = _img2img_pipe.to("cuda")

    # YOLO face detection
    if os.path.exists(FACE_MODEL_PATH):
        from ultralytics import YOLO
        _face_model = YOLO(FACE_MODEL_PATH)

    # Clear fragmentation from loading
    torch.cuda.empty_cache()

    t_load = time.time() - t0
    _pipe = pipe
    log.info("[load] Pipeline ready in %.1fs — LoRAs=%d, IP-Adapter=%s, FreeU=True, ADetailer=%s, VAE-slice=True",
             t_load, len(loaded), _ip_adapter_loaded, _face_model is not None)
    log.info("[load] VRAM after cleanup: %s", _vram_status())


# ── Generation ───────────────────────────────────────────────────────

def _generate(params):
    """Generate an image with detailed prompt logging and VRAM management."""
    global _gen_count

    prompt = params.get("prompt", "")
    steps = params.get("steps", 35)
    width = params.get("width", 832)
    height = params.get("height", 1216)
    cfg = params.get("cfg", 6.0)
    seed = params.get("seed")
    negative = params.get("negative")
    face_ref_b64 = params.get("face_refs_b64") or params.get("face_ref_b64")
    face_scale = params.get("face_scale", IP_ADAPTER_SCALE)
    mask_mode = params.get("mask_mode", "horizontal")

    if not prompt:
        return {"error": "Empty prompt"}

    _gen_count += 1
    gen_id = _gen_count

    # ── Log received params ──────────────────────────────────────────
    log.info("=" * 80)
    log.info("[gen #%d] NEW REQUEST", gen_id)
    log.info("[gen #%d] Resolution: %dx%d | Steps: %d | CFG: %.1f", gen_id, width, height, steps, cfg)
    log.info("[gen #%d] RECEIVED PROMPT: %s", gen_id, prompt)
    if negative:
        log.info("[gen #%d] RECEIVED NEGATIVE: %s", gen_id, negative)
    else:
        log.info("[gen #%d] RECEIVED NEGATIVE: (none — using default)", gen_id)
    log.info("[gen #%d] Face refs: %s | IP-Adapter loaded: %s | Mask mode: %s",
             gen_id, "yes (%d)" % len(face_ref_b64) if face_ref_b64 else "none",
             _ip_adapter_loaded, mask_mode)

    # ── Build positive prompt ────────────────────────────────────────
    full_prompt = PONY_QUALITY_PREFIX + prompt
    log.info("[gen #%d] FULL POSITIVE (prefix + prompt): %s", gen_id, full_prompt)

    # CLIP trim positive
    _tok = _pipe.tokenizer
    full_prompt, pos_tokens, pos_clipped = _clip_trim(full_prompt, "positive", _tok)
    if pos_clipped:
        log.info("[gen #%d] FINAL POSITIVE (after trim): %s", gen_id, full_prompt)

    # ── Build negative prompt ────────────────────────────────────────
    # Append custom negative to default (don't replace — default has critical anti-style tags)
    neg = DEFAULT_NEGATIVE + (", " + negative if negative else "")
    log.info("[gen #%d] FULL NEGATIVE: %s", gen_id, neg)

    # CLIP trim negative
    neg, neg_tokens, neg_clipped = _clip_trim(neg, "negative", _tok)
    if neg_clipped:
        log.info("[gen #%d] FINAL NEGATIVE (after trim): %s", gen_id, neg)

    # ── Face references ──────────────────────────────────────────────
    face_images = []
    if face_ref_b64 and _ip_adapter_loaded:
        if isinstance(face_ref_b64, str):
            face_ref_b64 = [face_ref_b64]
        for i, b64_str in enumerate(face_ref_b64):
            if b64_str:
                try:
                    img = _b64_to_pil(b64_str)
                    face_images.append(img)
                    log.info("[gen #%d] Face ref %d: %dx%d (%d bytes b64)", gen_id, i, img.width, img.height, len(b64_str))
                except Exception as e:
                    log.warning("[gen #%d] Face ref %d decode failed: %s", gen_id, i, e)
    elif face_ref_b64 and not _ip_adapter_loaded:
        log.warning("[gen #%d] Face refs provided but IP-Adapter NOT loaded — ignoring", gen_id)

    # Multi-character negative
    if len(face_images) >= 2:
        neg = neg + ", " + MULTI_CHAR_NEGATIVE
        log.info("[gen #%d] Multi-char negative added (%d faces)", gen_id, len(face_images))

    # Seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    log.info("[gen #%d] Seed: %d", gen_id, seed)

    # ── VRAM check before generation ─────────────────────────────────
    log.info("[gen #%d] VRAM before gen: %s", gen_id, _vram_status())

    with _pipe_lock:
        # Clear cache before generation to reduce fragmentation
        _torch.cuda.empty_cache()

        # IP-Adapter scale
        if _ip_adapter_loaded and face_images:
            _pipe.set_ip_adapter_scale(face_scale)
            log.info("[gen #%d] IP-Adapter scale: %.2f (%d face refs)", gen_id, face_scale, len(face_images))
        elif _ip_adapter_loaded:
            _pipe.set_ip_adapter_scale(0.0)

        generator = _torch.Generator("cuda").manual_seed(seed)

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

        # IP-Adapter image handling
        if _ip_adapter_loaded and not face_images:
            gen_kwargs["ip_adapter_image"] = _Image.new("RGB", (224, 224), (128, 128, 128))

        if face_images and _ip_adapter_loaded:
            if len(face_images) > 1:
                # Multi-face gradient masking
                import numpy as _np
                from PIL import ImageFilter as _IF

                _n = len(face_images)
                _gap = 0.08
                _blur_radius = max(width, height) // 12
                _masks = []

                for _i in range(_n):
                    _mask = _Image.new("L", (width, height), 0)
                    _slice = (1.0 - _gap * (_n - 1)) / _n
                    if mask_mode == "vertical":
                        _y0 = max(0, int(height * (_i * (_slice + _gap))))
                        _y1 = min(height, int(height * (_i * (_slice + _gap) + _slice)))
                        from PIL import ImageDraw as _ID
                        _ID.Draw(_mask).rectangle([0, _y0, width, _y1], fill=255)
                    else:
                        _x0 = max(0, int(width * (_i * (_slice + _gap))))
                        _x1 = min(width, int(width * (_i * (_slice + _gap) + _slice)))
                        from PIL import ImageDraw as _ID
                        _ID.Draw(_mask).rectangle([_x0, 0, _x1, height], fill=255)
                    _mask = _mask.filter(_IF.GaussianBlur(radius=_blur_radius))
                    _masks.append(_mask)

                import torchvision.transforms.functional as _TF
                _lh, _lw = height // 8, width // 8
                _mt = [_TF.to_tensor(_m.resize((_lw, _lh), _Image.BILINEAR)).squeeze(0) for _m in _masks]
                _processed = _torch.stack(_mt).unsqueeze(0)

                gen_kwargs["ip_adapter_image"] = [face_images]
                gen_kwargs["cross_attention_kwargs"] = {"ip_adapter_masks": [_processed.to("cuda")]}
                _pipe.set_ip_adapter_scale([[face_scale] * _n])
                log.info("[gen #%d] Multi-face masking: %d faces, mode=%s, scale=%.2f, blur=%d",
                         gen_id, _n, mask_mode, face_scale, _blur_radius)
            else:
                gen_kwargs["ip_adapter_image"] = face_images[0]
                log.info("[gen #%d] Single face ref, scale=%.2f", gen_id, face_scale)

        # Generate
        try:
            log.info("[gen #%d] Starting inference...", gen_id)
            image = _pipe(**gen_kwargs).images[0]
        except Exception as gen_err:
            t_gen = time.time() - t0
            log.error("[gen #%d] FAILED in %.1fs: %s", gen_id, t_gen, gen_err)
            log.error("[gen #%d] VRAM at failure: %s", gen_id, _vram_status())
            if _ip_adapter_loaded:
                _pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
            _torch.cuda.empty_cache()
            return {"error": "Generation failed: %s" % str(gen_err)}

        t_gen = time.time() - t0

        # Reset IP-Adapter scale
        if _ip_adapter_loaded and face_images and len(face_images) > 1:
            _pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)

        # Clear VRAM after generation
        _torch.cuda.empty_cache()

    # Encode result
    image_b64 = _pil_to_b64(image)

    log.info("[gen #%d] DONE in %.1fs | %dx%d | seed=%d | ip=%s | %d bytes",
             gen_id, t_gen, width, height, seed, bool(face_images), len(image_b64))
    log.info("[gen #%d] VRAM after gen: %s", gen_id, _vram_status())
    log.info("=" * 80)

    return {
        "image_b64": image_b64,
        "model": "cyberrealistic-pony-v8",
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "seed": seed,
        "gen_time": round(t_gen, 1),
        "ip_adapter": bool(face_images),
        "face_scale": face_scale if face_images else None,
        "freeu": True,
        "pos_tokens": pos_tokens,
        "neg_tokens": neg_tokens,
        "pos_clipped": pos_clipped,
    }


# ── ADetailer ────────────────────────────────────────────────────────

def _adetail(params):
    """ADetailer: detect faces, crop, img2img with face ref, Poisson-blend back."""
    import cv2
    import numpy as np

    image_b64 = params.get("image_b64", "")
    face_refs_b64 = params.get("face_refs_b64", [])
    face_scale = params.get("face_scale", 0.55)
    denoise = params.get("denoise", 0.25)
    steps = params.get("steps", 20)
    face_prompt = params.get("prompt")
    padding_mult = params.get("padding", 2.5)
    conf_thresh = params.get("conf", 0.40)

    if not image_b64:
        return {"error": "No image_b64 provided"}
    if not _face_model:
        return {"error": "Face detection model not loaded"}
    if not _img2img_pipe or not _ip_adapter_loaded:
        return {"error": "Img2Img pipeline or IP-Adapter not ready"}

    log.info("[adetail] Starting — denoise=%.2f, steps=%d, scale=%.2f, padding=%.1f, conf=%.2f",
             denoise, steps, face_scale, padding_mult, conf_thresh)

    src_image = _b64_to_pil(image_b64)
    w, h = src_image.size
    log.info("[adetail] Source image: %dx%d", w, h)

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
        return {"image_b64": _pil_to_b64(src_image), "faces_fixed": 0, "faces_detected": 0}

    faces.sort(key=lambda f: (f[0] + f[2]) / 2)
    log.info("[adetail] Detected %d faces: %s",
             len(faces), ", ".join("(%.0f,%.0f conf=%.2f)" % ((f[0]+f[2])/2, (f[1]+f[3])/2, f[4]) for f in faces))

    # Decode face references
    if isinstance(face_refs_b64, str):
        face_refs_b64 = [face_refs_b64]
    ref_images = []
    for b64_str in face_refs_b64:
        if b64_str:
            try:
                ref_images.append(_b64_to_pil(b64_str))
            except Exception:
                ref_images.append(None)
        else:
            ref_images.append(None)

    num_to_fix = min(len(faces), len(ref_images))
    log.info("[adetail] Will fix %d/%d faces (%d refs provided)", num_to_fix, len(faces), len(ref_images))

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

    fixed_count = 0
    errors = []
    current_cv = cv2.cvtColor(np.array(src_image), cv2.COLOR_RGB2BGR)

    with _pipe_lock:
        _torch.cuda.empty_cache()
        _img2img_pipe.to(dtype=_torch.float32)

        for i in range(num_to_fix):
            x1, y1, x2, y2, conf = faces[i]
            ref_img = ref_images[i]
            if ref_img is None:
                log.info("[adetail] Face %d: no ref, skipping", i)
                continue

            face_w, face_h = x2 - x1, y2 - y1
            pad_x = face_w * (padding_mult - 1) / 2
            pad_y = face_h * (padding_mult - 1) / 2
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y * 0.8))
            cx2 = min(w, int(x2 + pad_x))
            cy2 = min(h, int(y2 + pad_y * 1.2))
            crop_w, crop_h = cx2 - cx1, cy2 - cy1

            scale = max(512 / min(crop_w, crop_h), 1.0)
            target_w = min(768, max(512, ((int(crop_w * scale) + 7) // 8) * 8))
            target_h = min(768, max(512, ((int(crop_h * scale) + 7) // 8) * 8))

            current_pil = _Image.fromarray(cv2.cvtColor(current_cv, cv2.COLOR_BGR2RGB))
            face_crop = current_pil.crop((cx1, cy1, cx2, cy2)).resize((target_w, target_h), _Image.LANCZOS)

            _img2img_pipe.set_ip_adapter_scale(face_scale)
            gen = _torch.Generator("cuda").manual_seed(42 + i)

            log.info("[adetail] Face %d/%d: crop=(%d,%d,%d,%d) %dx%d->%dx%d denoise=%.2f conf=%.2f",
                     i+1, num_to_fix, cx1, cy1, cx2, cy2, crop_w, crop_h, target_w, target_h, denoise, conf)

            try:
                fixed_crop = _img2img_pipe(
                    prompt=face_prompt, negative_prompt=face_neg,
                    image=face_crop, ip_adapter_image=ref_img,
                    num_inference_steps=steps, guidance_scale=6.0,
                    strength=denoise, generator=gen,
                ).images[0]

                fixed_crop = fixed_crop.resize((crop_w, crop_h), _Image.LANCZOS)
                fixed_cv = cv2.cvtColor(np.array(fixed_crop), cv2.COLOR_RGB2BGR)

                mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                margin_x, margin_y = int(crop_w * 0.10), int(crop_h * 0.10)
                cv2.ellipse(mask, (crop_w//2, crop_h//2),
                            ((crop_w//2)-margin_x, (crop_h//2)-margin_y), 0, 0, 360, 255, -1)
                mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(3, crop_w * 0.03))
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                current_cv = cv2.seamlessClone(fixed_cv, current_cv, mask,
                                                (cx1 + crop_w//2, cy1 + crop_h//2), cv2.NORMAL_CLONE)
                fixed_count += 1
                log.info("[adetail] Face %d: Poisson-blended OK", i+1)

            except Exception as e:
                log.error("[adetail] Face %d FAILED: %s", i, e, exc_info=True)
                errors.append(str(e))

        _img2img_pipe.to(dtype=_torch.float16)
        _img2img_pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
        _torch.cuda.empty_cache()

    result_image = _Image.fromarray(cv2.cvtColor(current_cv, cv2.COLOR_BGR2RGB))
    log.info("[adetail] DONE — fixed %d/%d faces", fixed_count, len(faces))

    return {
        "image_b64": _pil_to_b64(result_image),
        "faces_detected": len(faces),
        "faces_fixed": fixed_count,
        "face_scale": face_scale,
        "denoise": denoise,
        "steps": steps,
        "errors": errors if errors else None,
    }


# ── RunPod Handler ───────────────────────────────────────────────────

def handler(job):
    """RunPod serverless handler."""
    job_input = job.get("input", {})
    action = job_input.get("action", "generate")

    if _pipe is None:
        return {"error": "Pipeline still loading, please retry"}

    log.info("[handler] Action: %s | Job keys: %s", action, sorted(job_input.keys()))

    try:
        if action == "adetail":
            return _adetail(job_input)
        else:
            return _generate(job_input)
    except Exception as e:
        log.error("[handler] Unhandled error: %s", e, exc_info=True)
        log.error("[handler] VRAM at crash: %s", _vram_status())
        _torch.cuda.empty_cache()
        return {"error": str(e)}


# ── Startup ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 80)
    log.info("SDXL Worker starting...")
    log.info("Model volume: %s", _VOL)
    log.info("PYTORCH_CUDA_ALLOC_CONF: %s", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(not set)"))
    _ensure_models()
    _load_pipeline()
    log.info("Starting RunPod serverless worker...")
    log.info("=" * 80)
    runpod.serverless.start({"handler": handler})
