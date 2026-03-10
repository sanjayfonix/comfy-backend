"""
Comify AI - Virtual Try-On & Size Recommendation Backend
=========================================================
FastAPI backend that provides:
  POST /api/generate-tryon      -> Virtual try-on image generation
  POST /api/size-recommendation  -> Body measurement & size recommendation

Designed for demo usage. Supports two modes:
  1. LOCAL mode  – uses MediaPipe for pose/segmentation + PIL compositing
  2. COLAB mode  – forwards heavy inference to a Google Colab GPU runtime via ngrok
"""

import os
import io
import uuid
import base64
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

# Ensure .env is loaded from the backend directory regardless of cwd
_backend_dir = Path(__file__).resolve().parent
load_dotenv(_backend_dir / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COLAB_ENDPOINT = os.getenv("COLAB_ENDPOINT", "")  # e.g. https://xxxx.ngrok-free.app
USE_COLAB = bool(COLAB_ENDPOINT)
N8N_WEBHOOK_BASE = os.getenv("N8N_WEBHOOK_BASE", "")  # e.g. http://localhost:5678/webhook
OUTPUT_DIR = _backend_dir / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# HuggingFace Space – call IDM-VTON without any local GPU or Colab
HF_SPACE_ID = os.getenv("HF_SPACE_ID", "yisol/IDM-VTON")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # optional HF access token
USE_HF_SPACE = os.getenv("USE_HF_SPACE", "true").lower() in ("1", "true", "yes")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comify-ai")
logger.info(f"Backend dir: {_backend_dir}")
logger.info(f"HF_TOKEN loaded: {'yes (' + HF_TOKEN[:10] + '...)' if HF_TOKEN else 'NO'}")
logger.info(f"USE_HF_SPACE: {USE_HF_SPACE}")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Comify AI – Virtual Try-On API",
    version="1.0.0",
    description="Demo backend for virtual try-on and size recommendation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/api/debug-hf")
async def debug_hf():
    """Debug endpoint: test HF Space connectivity and token."""
    info = {
        "USE_HF_SPACE": USE_HF_SPACE,
        "HF_SPACE_ID": HF_SPACE_ID,
        "HF_TOKEN_SET": bool(HF_TOKEN),
        "HF_TOKEN_PREFIX": HF_TOKEN[:10] + "..." if HF_TOKEN else "NONE",
        "backend_dir": str(_backend_dir),
    }
    if USE_HF_SPACE and HF_TOKEN:
        try:
            client = _get_hf_client()
            info["client_connected"] = True
            info["client_src"] = str(getattr(client, 'src', 'unknown'))
        except Exception as e:
            info["client_error"] = f"{type(e).__name__}: {e}"
    return info


# ---------------------------------------------------------------------------
# Lazy-loaded MediaPipe (Tasks API – mediapipe >= 0.10.x)
# ---------------------------------------------------------------------------
_mp_pose = None
_mp_segmenter = None

MODELS_DIR = Path(__file__).parent / "models"


def get_mediapipe_pose():
    """Return a PoseLandmarker instance (Tasks API)."""
    global _mp_pose
    if _mp_pose is None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        model_path = str(MODELS_DIR / "pose_landmarker_heavy.task")
        if not Path(model_path).exists():
            raise RuntimeError(
                f"Pose model not found at {model_path}. "
                "Download it from https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            )

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _mp_pose = vision.PoseLandmarker.create_from_options(options)
        logger.info("✓ PoseLandmarker loaded (Tasks API)")
    return _mp_pose


def get_mediapipe_segmenter():
    """Return an ImageSegmenter instance (Tasks API)."""
    global _mp_segmenter
    if _mp_segmenter is None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        model_path = str(MODELS_DIR / "selfie_multiclass_256x256.tflite")
        if not Path(model_path).exists():
            raise RuntimeError(
                f"Segmenter model not found at {model_path}. "
                "Download it from https://storage.googleapis.com/mediapipe-models/"
                "image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
            )

        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        _mp_segmenter = vision.ImageSegmenter.create_from_options(options)
        logger.info("✓ ImageSegmenter loaded (Tasks API)")
    return _mp_segmenter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def decode_base64_image(data: str) -> Image.Image:
    """Decode a data-URI or raw base64 string into a PIL Image."""
    if "," in data:
        data = data.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def image_to_base64(img: Image.Image, fmt: str="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def save_image(img: Image.Image, prefix: str="tryon") -> str:
    """Save image to outputs/ and return the relative URL path."""
    fname = f"{prefix}_{uuid.uuid4().hex[:12]}.jpg"
    fpath = OUTPUT_DIR / fname
    img.save(fpath, "JPEG", quality=92)
    return f"/outputs/{fname}"


def pil_from_upload(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")

# ---------------------------------------------------------------------------
# Size Recommendation Logic
# ---------------------------------------------------------------------------


SIZE_CHART = {
    # (min_chest, max_chest, min_waist, max_waist) -> size
    "XS": {"chest": (76, 84), "waist": (60, 68)},
    "S": {"chest": (84, 92), "waist": (68, 76)},
    "M": {"chest": (92, 100), "waist": (76, 84)},
    "L": {"chest": (100, 108), "waist": (84, 92)},
    "XL": {"chest": (108, 116), "waist": (92, 100)},
    "XXL": {"chest": (116, 130), "waist": (100, 115)},
}


def estimate_measurements_from_landmarks(landmarks, img_width: int, img_height: int) -> dict:
    """
    Estimate body measurements from MediaPipe Pose landmarks.
    Uses pixel distances + a heuristic scale factor based on average human proportions.
    """

    def lm(idx):
        l = landmarks[idx]
        return np.array([l.x * img_width, l.y * img_height])

    # Key landmark indices (MediaPipe Pose)
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    NOSE = 0

    left_shoulder = lm(LEFT_SHOULDER)
    right_shoulder = lm(RIGHT_SHOULDER)
    left_hip = lm(LEFT_HIP)
    right_hip = lm(RIGHT_HIP)
    nose = lm(NOSE)
    left_ankle = lm(LEFT_ANKLE)
    right_ankle = lm(RIGHT_ANKLE)

    # Pixel distances
    shoulder_px = np.linalg.norm(left_shoulder - right_shoulder)
    hip_px = np.linalg.norm(left_hip - right_hip)
    torso_height_px = np.linalg.norm(
        (left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2
    )
    body_height_px = np.linalg.norm(nose - (left_ankle + right_ankle) / 2)

    # Heuristic: assume average height ~170cm if no reference
    # Scale factor = real_cm / pixel
    assumed_height_cm = 170.0
    scale = assumed_height_cm / max(body_height_px, 1)

    shoulder_cm = round(shoulder_px * scale, 1)
    hip_cm = round(hip_px * scale, 1)

    # Chest ≈ shoulder_width * π / 2 (ellipse approximation)
    chest_cm = round(shoulder_cm * 2.6, 1)
    # Waist ≈ midpoint between shoulder and hip width * π / 2
    waist_width = (shoulder_px + hip_px) / 2
    waist_cm = round(waist_width * scale * 2.3, 1)
    # Hip circumference
    hip_circ_cm = round(hip_cm * 2.8, 1)

    height_cm = round(body_height_px * scale, 1)

    return {
        "shoulders": shoulder_cm,
        "chest": chest_cm,
        "waist": waist_cm,
        "hips": hip_circ_cm,
        "height": height_cm,
    }


def recommend_size(measurements: dict) -> str:
    """Map measurements to the closest size using the size chart."""
    chest = measurements.get("chest", 92)
    waist = measurements.get("waist", 78)

    best_size = "M"
    best_score = float("inf")

    for size, ranges in SIZE_CHART.items():
        chest_mid = sum(ranges["chest"]) / 2
        waist_mid = sum(ranges["waist"]) / 2
        score = abs(chest - chest_mid) + abs(waist - waist_mid)
        if score < best_score:
            best_score = score
            best_size = size

    return best_size


def recommend_size_per_category(measurements: dict) -> dict:
    """Return size recommendations per clothing category."""
    base = recommend_size(measurements)
    sizes = list(SIZE_CHART.keys())
    idx = sizes.index(base) if base in sizes else 2

    return {
        "tops": base,
        "bottoms": sizes[min(idx + 1, len(sizes) - 1)] if measurements.get("hips", 0) > 100 else base,
        "dresses": base,
    }

# ---------------------------------------------------------------------------
# Pose Detection & Landmark Extraction
# ---------------------------------------------------------------------------


def detect_pose(image: Image.Image) -> dict:
    """Run MediaPipe PoseLandmarker (Tasks API) and return structured landmarks + measurements."""
    import mediapipe as mp

    img_array = np.array(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

    pose_landmarker = get_mediapipe_pose()
    results = pose_landmarker.detect(mp_image)

    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        raise HTTPException(
            status_code=422,
            detail="Could not detect a person in the image. Please upload a clear full-body photo.",
        )

    # Tasks API: results.pose_landmarks is List[List[NormalizedLandmark]]
    landmarks = results.pose_landmarks[0]  # first detected person
    w, h = image.size

    # Extract named landmarks for the frontend
    MP_LANDMARK_NAMES = {
        0: "nose", 11: "left_shoulder", 12: "right_shoulder",
        13: "left_elbow", 14: "right_elbow",
        15: "left_wrist", 16: "right_wrist",
        23: "left_hip", 24: "right_hip",
        25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle",
    }

    # Synthesize "neck" as midpoint of shoulders
    ls = landmarks[11]
    rs = landmarks[12]
    neck_x = (ls.x + rs.x) / 2
    neck_y = (ls.y + rs.y) / 2 - 0.02

    # Synthesize "waist" points as midpoint between shoulders and hips
    lh = landmarks[23]
    rh = landmarks[24]
    lw_x = (ls.x + lh.x) / 2
    lw_y = (ls.y + lh.y) / 2
    rw_x = (rs.x + rh.x) / 2
    rw_y = (rs.y + rh.y) / 2

    frontend_landmarks = [{"name": "neck", "x": round(neck_x, 4), "y": round(neck_y, 4)}]
    frontend_landmarks.append({"name": "left_waist", "x": round(lw_x, 4), "y": round(lw_y, 4)})
    frontend_landmarks.append({"name": "right_waist", "x": round(rw_x, 4), "y": round(rw_y, 4)})

    for idx, name in MP_LANDMARK_NAMES.items():
        lm = landmarks[idx]
        frontend_landmarks.append({
            "name": name,
            "x": round(lm.x, 4),
            "y": round(lm.y, 4),
        })

    measurements = estimate_measurements_from_landmarks(landmarks, w, h)
    size = recommend_size(measurements)
    category_sizes = recommend_size_per_category(measurements)

    # Get visibility/confidence from the nose landmark
    nose_vis = getattr(landmarks[0], "visibility", 0.95)
    confidence = round(min(max(nose_vis, 0.85), 0.99), 4)

    return {
        "bodyDetected": True,
        "confidence": confidence,
        "bodyMeasurements": measurements,
        "poseDetection": {
            "detected": True,
            "pose": "standing",
            "landmarks": frontend_landmarks,
        },
        "recommendedSize": size,
        "clothingFitPrediction": category_sizes,
    }

# ---------------------------------------------------------------------------
# Virtual Try-On (Local composite mode)
# ---------------------------------------------------------------------------


def composite_tryon(person_img: Image.Image, garment_img: Image.Image, landmarks: list) -> Image.Image:
    """
    Demo-level virtual try-on using PIL compositing.
    Warps and overlays the garment onto the person image based on detected landmarks.
    """
    person = person_img.copy()
    pw, ph = person.size

    # Parse landmarks
    lm_dict = {l["name"]: l for l in landmarks}

    def pt(name):
        lm = lm_dict.get(name)
        if lm:
            return (int(lm["x"] * pw), int(lm["y"] * ph))
        return None

    left_shoulder = pt("left_shoulder")
    right_shoulder = pt("right_shoulder")
    neck = pt("neck")
    left_hip = pt("left_hip")
    right_hip = pt("right_hip")

    if not all([left_shoulder, right_shoulder, neck, left_hip, right_hip]):
        # Fallback: center overlay
        garment_resized = garment_img.resize((pw // 2, ph // 2))
        person.paste(garment_resized, (pw // 4, ph // 4), garment_resized if garment_resized.mode == "RGBA" else None)
        return person

    # Calculate placement
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    torso_height = abs(((left_hip[1] + right_hip[1]) // 2) - neck[1])

    garment_width = int(shoulder_width * 2.2)
    garment_height = int(torso_height * 1.5)

    center_x = (left_shoulder[0] + right_shoulder[0]) // 2
    top_y = neck[1] + int(torso_height * 0.05)

    # Resize garment
    garment_resized = garment_img.resize((garment_width, garment_height), Image.LANCZOS)

    # Make background transparent if not already
    if garment_resized.mode != "RGBA":
        garment_resized = garment_resized.convert("RGBA")
        data = np.array(garment_resized)
        # Simple background removal: remove near-white and near-black pixels
        r, g, b, a = data[:,:, 0], data[:,:, 1], data[:,:, 2], data[:,:, 3]
        white_mask = (r > 240) & (g > 240) & (b > 240)
        light_mask = (r > 220) & (g > 220) & (b > 220) & (np.abs(r.astype(int) - g.astype(int)) < 15)
        bg_mask = white_mask | light_mask
        data[bg_mask, 3] = 0
        # Feather edges
        garment_resized = Image.fromarray(data)
        # Apply slight blur to alpha for smooth edges
        alpha = garment_resized.split()[3]
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=2))
        garment_resized.putalpha(alpha)

    # Calculate paste position
    paste_x = center_x - garment_width // 2
    paste_y = top_y

    # Ensure within bounds
    paste_x = max(0, min(paste_x, pw - garment_width))
    paste_y = max(0, min(paste_y, ph - garment_height))

    # Create body-area mask to constrain garment
    body_mask = Image.new("L", person.size, 0)
    draw = ImageDraw.Draw(body_mask)

    body_poly = [
        (left_shoulder[0] - int(shoulder_width * 0.3), neck[1] + 10),
        (right_shoulder[0] + int(shoulder_width * 0.3), neck[1] + 10),
        (right_hip[0] + int(shoulder_width * 0.3), right_hip[1] + 30),
        (left_hip[0] - int(shoulder_width * 0.3), left_hip[1] + 30),
    ]
    draw.polygon(body_poly, fill=255)
    body_mask = body_mask.filter(ImageFilter.GaussianBlur(radius=8))

    # Composite: paste garment onto person
    person_rgba = person.convert("RGBA")
    overlay = Image.new("RGBA", person.size, (0, 0, 0, 0))

    # Adjust garment brightness/contrast to match person
    enhancer = ImageEnhance.Brightness(garment_resized)
    garment_resized = enhancer.enhance(0.95)
    enhancer = ImageEnhance.Contrast(garment_resized)
    garment_resized = enhancer.enhance(1.05)

    overlay.paste(garment_resized, (paste_x, paste_y), garment_resized)

    # Apply body mask to overlay
    overlay_array = np.array(overlay)
    mask_array = np.array(body_mask)

    # Crop mask to overlay region
    for y in range(overlay_array.shape[0]):
        for x in range(overlay_array.shape[1]):
            if overlay_array[y, x, 3] > 0:
                mask_val = mask_array[y, x] / 255.0
                overlay_array[y, x, 3] = int(overlay_array[y, x, 3] * mask_val)

    overlay = Image.fromarray(overlay_array)

    # Final composite with blending
    result = Image.alpha_composite(person_rgba, overlay)
    return result.convert("RGB")

# ---------------------------------------------------------------------------
# HuggingFace Spaces – IDM-VTON via Gradio Client (no GPU needed)
# ---------------------------------------------------------------------------


_hf_client = None

# Map product categories to descriptive garment text for CLIP text encoding
GARMENT_DESC_MAP = {
    "tops": "Short Sleeve Round Neck T-shirt",
    "t-shirts": "Short Sleeve Round Neck T-shirt",
    "shirts": "Long Sleeve Button-Up Shirt",
    "blouses": "Elegant Blouse Top",
    "hoodies": "Pullover Hoodie Sweatshirt",
    "sweaters": "Knit Pullover Sweater",
    "jackets": "Casual Zip-Up Jacket",
    "coats": "Long Coat Outerwear",
    "dresses": "Casual Dress",
    "pants": "Straight Leg Trousers",
    "jeans": "Denim Jeans",
    "shorts": "Casual Shorts",
    "skirts": "A-Line Skirt",
    "suits": "Formal Suit Jacket",
    "activewear": "Athletic Performance Top",
}


def _category_to_garment_desc(category: str) -> str:
    """Convert product category to a descriptive garment text for IDM-VTON."""
    return GARMENT_DESC_MAP.get(category.lower().strip(), "Short Sleeve Round Neck T-shirt")


def _preprocess_person_for_tryon(img: Image.Image) -> Image.Image:
    """
    Preprocess person image for IDM-VTON.
    - Converts to RGB
    - Pads to 3:4 aspect ratio (matching IDM-VTON's 768×1024)
    - Resizes to 768×1024 using high-quality LANCZOS resampling
    """
    img = img.convert("RGB")
    w, h = img.size
    target_ratio = 3 / 4  # width / height = 768 / 1024
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Image is too wide → pad top/bottom
        new_h = int(w / target_ratio)
        padded = Image.new("RGB", (w, new_h), (255, 255, 255))
        padded.paste(img, (0, (new_h - h) // 2))
    elif current_ratio < target_ratio:
        # Image is too tall → pad left/right
        new_w = int(h * target_ratio)
        padded = Image.new("RGB", (new_w, h), (255, 255, 255))
        padded.paste(img, ((new_w - w) // 2, 0))
    else:
        padded = img

    # Resize to IDM-VTON's native resolution
    result = padded.resize((768, 1024), Image.LANCZOS)
    logger.info(f"Person image preprocessed: {w}×{h} → 768×1024")
    return result


def _preprocess_garment_for_tryon(img: Image.Image) -> Image.Image:
    """
    Preprocess garment image for IDM-VTON.
    - Converts to RGB
    - Resizes to 768×1024 using high-quality LANCZOS resampling
    """
    img = img.convert("RGB")
    w, h = img.size
    result = img.resize((768, 1024), Image.LANCZOS)
    logger.info(f"Garment image preprocessed: {w}×{h} → 768×1024")
    return result


def _get_hf_client():
    """Lazy-init a Gradio Client connected to the IDM-VTON HuggingFace Space."""
    global _hf_client
    if _hf_client is None:
        from gradio_client import Client
        _hf_client = Client(HF_SPACE_ID, token=HF_TOKEN or None)
        logger.info(f"\u2713 Connected to HuggingFace Space: {HF_SPACE_ID}")
    return _hf_client


async def call_hf_space_tryon(
    person_img: Image.Image,
    garment_img: Image.Image,
    garment_desc: str="Short Sleeve Round Neck T-shirt",
    category: str="tops",
    num_steps: int=40,
    seed: int=42,
) -> Image.Image:
    """
    Call the IDM-VTON HuggingFace Space via the Gradio Client.
    No local GPU or Colab needed – runs entirely on HF infrastructure.

    Quality improvements:
    - Preprocesses person image to 768×1024 with proper 3:4 padding
    - Preprocesses garment image to 768×1024 with LANCZOS downsampling
    - Saves temp files as lossless PNG to avoid compression artifacts
    - Uses is_checked_crop=True for better crop & resize handling
    - Uses descriptive garment text for better CLIP text encoding
    - Defaults to 40 denoising steps for higher quality
    """
    import asyncio
    import tempfile
    from gradio_client import handle_file

    # Resolve garment description from category if generic
    if garment_desc in ("garment", "", None):
        garment_desc = _category_to_garment_desc(category)

    logger.info(f"IDM-VTON params: desc='{garment_desc}', steps={num_steps}, seed={seed}")

    # Preprocess images to optimal resolution for IDM-VTON
    person_img = _preprocess_person_for_tryon(person_img)
    garment_img = _preprocess_garment_for_tryon(garment_img)

    # gradio_client needs file paths — on Windows we must close the handle
    # before other processes can read the file
    person_fd, person_path = tempfile.mkstemp(suffix=".png")
    garment_fd, garment_path = tempfile.mkstemp(suffix=".png")
    os.close(person_fd)
    os.close(garment_fd)
    try:
        # Save as lossless PNG to avoid compression artifacts
        person_img.save(person_path, "PNG")
        garment_img.save(garment_path, "PNG")

        client = _get_hf_client()

        # Run the blocking gradio predict() in a thread so we don't block the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: client.predict(
                dict={"background": handle_file(person_path), "layers": [], "composite": None},
                garm_img=handle_file(garment_path),
                garment_des=garment_desc,
                is_checked=True,  # auto-generate mask via OpenPose + human parsing
                is_checked_crop=True,  # auto-crop to 3:4 ratio for best results
                denoise_steps=num_steps,
                seed=seed,
                api_name="/tryon",
            ),
        )

        # Gradio returns a file path (or tuple of paths) for image outputs
        if isinstance(result, (list, tuple)):
            result_path = result[0]
        else:
            result_path = result
        return Image.open(result_path).convert("RGB")
    finally:
        try:
            os.unlink(person_path)
        except OSError:
            pass
        try:
            os.unlink(garment_path)
        except OSError:
            pass

# ---------------------------------------------------------------------------
# Google Colab Proxy (for real AI inference)
# ---------------------------------------------------------------------------


async def call_colab_tryon(person_b64: str, garment_b64: str,
                          category: str="tops", num_steps: int=30) -> dict:
    """Forward try-on request to Google Colab running IDM-VTON."""
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            f"{COLAB_ENDPOINT}/generate-tryon",
            json={
                "person_image": person_b64,
                "garment_image": garment_b64,
                "category": category,
                "num_steps": num_steps,
                "guidance_scale": 2.5,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def call_colab_size(person_b64: str) -> dict:
    """Forward size recommendation request to Colab."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{COLAB_ENDPOINT}/size-recommendation",
            json={"person_image": person_b64},
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# n8n Webhook Proxy
# ---------------------------------------------------------------------------
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")


async def notify_n8n(event: str, payload: dict):
    """Fire-and-forget notification to n8n workflow."""
    if not N8N_WEBHOOK_URL:
        return
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(N8N_WEBHOOK_URL, json={"event": event, **payload})
    except Exception as e:
        logger.warning(f"n8n notification failed: {e}")

# ============================= API ENDPOINTS ==============================


@app.get("/api/health")
async def health():
    if USE_HF_SPACE:
        mode = "hf-space+idm-vton"
        tryon_model = f"IDM-VTON (HF Space: {HF_SPACE_ID})"
    elif USE_COLAB:
        mode = "colab+idm-vton"
        tryon_model = "IDM-VTON"
    else:
        mode = "local"
        tryon_model = "local-composite"
    return {
        "status": "ok",
        "mode": mode,
        "hf_space": HF_SPACE_ID if USE_HF_SPACE else None,
        "colab_endpoint": COLAB_ENDPOINT if USE_COLAB else None,
        "n8n_configured": bool(N8N_WEBHOOK_BASE),
        "tryon_model": tryon_model,
        "size_model": "MediaPipe-Pose",
    }


class SizeRecommendationResponse(BaseModel):
    bodyDetected: bool
    confidence: float
    bodyMeasurements: dict
    poseDetection: dict
    recommendedSize: str
    clothingFitPrediction: dict


@app.post("/api/size-recommendation", response_model=SizeRecommendationResponse)
async def size_recommendation(person_image: UploadFile=File(...)):
    """
    Analyze an uploaded person photo and return body measurements + size recommendation.
    """
    start = time.time()
    logger.info("📏 Size recommendation request received")

    img_bytes = await person_image.read()
    img = pil_from_upload(img_bytes)

    if USE_COLAB:
        try:
            b64 = image_to_base64(img)
            result = await call_colab_size(b64)
            await notify_n8n("size_recommendation", {"duration_ms": int((time.time() - start) * 1000)})
            return result
        except Exception as e:
            logger.warning(f"Colab call failed, falling back to local: {e}")

    # Local mode: MediaPipe pose detection
    result = detect_pose(img)

    duration = int((time.time() - start) * 1000)
    logger.info(f"📏 Size recommendation completed in {duration}ms -> {result['recommendedSize']}")
    await notify_n8n("size_recommendation", {
        "recommended_size": result["recommendedSize"],
        "measurements": result["bodyMeasurements"],
        "duration_ms": duration,
    })

    return result


class TryOnResponse(BaseModel):
    try_on_image_url: str
    recommended_size: str
    bodyMeasurements: Optional[dict] = None
    confidence: Optional[float] = None


@app.post("/api/generate-tryon", response_model=TryOnResponse)
async def generate_tryon(
    person_image: UploadFile=File(...),
    garment_image: UploadFile=File(...),
    category: str=Form("tops"),
):
    """
    Generate a virtual try-on image.
    Accepts person photo + garment image, returns the composited result.
    """
    start = time.time()
    logger.info(f"👔 Try-on request: category={category}")

    person_bytes = await person_image.read()
    garment_bytes = await garment_image.read()

    person_img = pil_from_upload(person_bytes)
    garment_img = pil_from_upload(garment_bytes)

    # ── Priority 1: HuggingFace Space (no GPU needed) ──
    if USE_HF_SPACE:
        try:
            logger.info(f"\U0001f680 Calling HF Space {HF_SPACE_ID} for try-on...")
            result_img = await call_hf_space_tryon(
                person_img, garment_img,
                category=category,
            )
            url = save_image(result_img, "tryon")

            # Also do local pose detection for size recommendation
            pose_data = detect_pose(person_img)
            duration = int((time.time() - start) * 1000)
            logger.info(f"\U0001f680 HF Space try-on completed in {duration}ms")
            return TryOnResponse(
                try_on_image_url=url,
                recommended_size=pose_data["recommendedSize"],
                bodyMeasurements=pose_data["bodyMeasurements"],
                confidence=pose_data["confidence"],
            )
        except Exception as e:
            logger.warning(f"HF Space try-on failed, falling back: {e}")

    # ── Priority 2: Google Colab GPU ──
    if USE_COLAB:
        try:
            person_b64 = image_to_base64(person_img)
            garment_b64 = image_to_base64(garment_img)
            colab_result = await call_colab_tryon(person_b64, garment_b64)

            # Save the result image locally
            result_img = decode_base64_image(colab_result["try_on_image"])
            url = save_image(result_img, "tryon")

            await notify_n8n("tryon_generated", {
                "image_url": url,
                "category": category,
                "duration_ms": int((time.time() - start) * 1000),
            })

            return TryOnResponse(
                try_on_image_url=url,
                recommended_size=colab_result.get("recommended_size", "M"),
                bodyMeasurements=colab_result.get("bodyMeasurements"),
                confidence=colab_result.get("confidence"),
            )
        except Exception as e:
            logger.warning(f"Colab try-on failed, falling back to local: {e}")

    # ── Priority 3: Local composite fallback ──
    # Local mode: MediaPipe + PIL composite
    pose_data = detect_pose(person_img)
    landmarks = pose_data["poseDetection"]["landmarks"]

    result_img = composite_tryon(person_img, garment_img, landmarks)
    url = save_image(result_img, "tryon")

    duration = int((time.time() - start) * 1000)
    logger.info(f"👔 Try-on generated in {duration}ms -> {url}")

    await notify_n8n("tryon_generated", {
        "image_url": url,
        "category": category,
        "recommended_size": pose_data["recommendedSize"],
        "duration_ms": duration,
    })

    return TryOnResponse(
        try_on_image_url=url,
        recommended_size=pose_data["recommendedSize"],
        bodyMeasurements=pose_data["bodyMeasurements"],
        confidence=pose_data["confidence"],
    )

# ---------------------------------------------------------------------------
# Base64 upload alternative (for n8n / programmatic access)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Base64 size recommendation (for n8n JSON body calls)
# ---------------------------------------------------------------------------


class Base64SizeRequest(BaseModel):
    person_image: str  # base64


@app.post("/api/size-recommendation-b64")
async def size_recommendation_base64(req: Base64SizeRequest):
    """Size recommendation accepting base64 image (for n8n integration)."""
    start = time.time()
    logger.info("Size recommendation (b64) request")

    img = decode_base64_image(req.person_image)

    if USE_COLAB:
        try:
            result = await call_colab_size(req.person_image)
            return result
        except Exception as e:
            logger.warning(f"Colab size failed, using local: {e}")

    result = detect_pose(img)
    duration = int((time.time() - start) * 1000)
    logger.info(f"Size (b64) done in {duration}ms -> {result['recommendedSize']}")
    return result

# ---------------------------------------------------------------------------
# Base64 try-on (for n8n JSON body calls)
# ---------------------------------------------------------------------------


class Base64TryOnRequest(BaseModel):
    person_image: str  # base64
    garment_image: str  # base64
    category: str = "tops"


@app.post("/api/generate-tryon-b64")
async def generate_tryon_base64(req: Base64TryOnRequest):
    """Try-on endpoint accepting base64 images (for n8n integration)."""
    start = time.time()

    person_img = decode_base64_image(req.person_image)
    garment_img = decode_base64_image(req.garment_image)

    # ── Priority 1: HuggingFace Space ──
    if USE_HF_SPACE:
        try:
            logger.info(f"\U0001f680 Calling HF Space {HF_SPACE_ID} for try-on (b64)...")
            result_img = await call_hf_space_tryon(
                person_img, garment_img,
                category=req.category,
            )
            url = save_image(result_img, "tryon")
            pose_data = detect_pose(person_img)
            return {
                "try_on_image_url": url,
                "try_on_image_b64": image_to_base64(result_img),
                "recommended_size": pose_data["recommendedSize"],
                "bodyMeasurements": pose_data["bodyMeasurements"],
                "model": f"IDM-VTON (HF Space: {HF_SPACE_ID})",
            }
        except Exception as e:
            logger.warning(f"HF Space (b64) failed, falling back: {e}")

    # ── Priority 2: Google Colab ──
    if USE_COLAB:
        try:
            colab_result = await call_colab_tryon(req.person_image, req.garment_image)
            result_img = decode_base64_image(colab_result["try_on_image"])
            url = save_image(result_img, "tryon")
            return {
                "try_on_image_url": url,
                "try_on_image_b64": colab_result["try_on_image"],
                "recommended_size": colab_result.get("recommended_size", "M"),
            }
        except Exception as e:
            logger.warning(f"Colab failed, using local: {e}")

    pose_data = detect_pose(person_img)
    result_img = composite_tryon(person_img, garment_img, pose_data["poseDetection"]["landmarks"])
    url = save_image(result_img, "tryon")

    return {
        "try_on_image_url": url,
        "try_on_image_b64": image_to_base64(result_img),
        "recommended_size": pose_data["recommendedSize"],
        "bodyMeasurements": pose_data["bodyMeasurements"],
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
