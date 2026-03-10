"""
Quick test: send a real try-on request to verify HuggingFace Space is actually called.
Uses an IDM-VTON example person image + the JColeFront garment.
"""
import base64, io, time, httpx
from pathlib import Path
from PIL import Image

API = "http://localhost:8001"

# Use a high-quality IDM-VTON example person image (768x1024)
person_path = r"C:\Users\Admin\Downloads\ComifyAiProject-main\ComifyAiProject-main\backend\test_images\00034_00.jpg"
if not Path(person_path).exists():
    # Fallback to old test image
    person_path = r"C:\Users\Admin\Downloads\ComifyAiProject-main\ComifyAiProject-main\backend\test_images\test_person.jpg"
    print("WARNING: Using low-res fallback person image. Run download_test_person.py first!")

with open(person_path, "rb") as f:
    person_b64 = base64.b64encode(f.read()).decode()
img = Image.open(person_path)
print(f"Person image: {img.size}, {len(person_b64)} chars b64")

# Load garment image
garment_path = r"C:\Users\Admin\Downloads\ComifyAiProject-main\ComifyAiProject-main\public\JColeFront.webp"
with open(garment_path, "rb") as f:
    garment_b64 = base64.b64encode(f.read()).decode()

print(f"Person image: {len(person_b64)} chars (base64)")
print(f"Garment image: {len(garment_b64)} chars (base64)")
print()

# -- Test 1: Size recommendation (local MediaPipe) --
print("=" * 50)
print("TEST 1: Size recommendation (local MediaPipe)")
print("=" * 50)
try:
    t0 = time.time()
    r = httpx.post(f"{API}/api/size-recommendation-b64", json={"person_image": person_b64}, timeout=30)
    dt = round(time.time() - t0, 2)
    print(f"  Status: {r.status_code} ({dt}s)")
    if r.status_code == 200:
        data = r.json()
        print(f"  Body detected: {data.get('bodyDetected')}")
        print(f"  Size: {data.get('recommendedSize')}")
    else:
        print(f"  Error: {r.text[:300]}")
except Exception as e:
    print(f"  Exception: {e}")

print()

# -- Test 2: Try-on via HF Space --
print("=" * 50)
print("TEST 2: Virtual Try-On (HuggingFace Space)")
print("  This calls yisol/IDM-VTON on HF infrastructure.")
print("  First call may take 1-3 minutes (cold start + inference).")
print("=" * 50)
try:
    t0 = time.time()
    print(f"  Sending request at {time.strftime('%H:%M:%S')}...")
    r = httpx.post(
        f"{API}/api/generate-tryon-b64",
        json={
            "person_image": person_b64,
            "garment_image": garment_b64,
            "category": "tops",
        },
        timeout=300,  # 5 min timeout for cold start
    )
    dt = round(time.time() - t0, 2)
    print(f"  Status: {r.status_code} ({dt}s)")
    if r.status_code == 200:
        data = r.json()
        has_url = bool(data.get("try_on_image_url"))
        has_b64 = bool(data.get("try_on_image_b64"))
        model = data.get("model", "unknown")
        size = data.get("recommended_size", "?")
        print(f"  SUCCESS!")
        print(f"  Model used: {model}")
        print(f"  Has result image URL: {has_url}")
        print(f"  Has result image B64: {has_b64}")
        print(f"  Recommended size: {size}")
        if has_b64:
            # Verify it's a valid image
            img_data = base64.b64decode(data["try_on_image_b64"])
            img = Image.open(io.BytesIO(img_data))
            print(f"  Result image size: {img.size}")
            img.save(r"C:\Users\Admin\Downloads\ComifyAiProject-main\ComifyAiProject-main\backend\outputs\hf_test_result.png")
            print(f"  Saved to: outputs/hf_test_result.png")
    else:
        print(f"  FAILED: {r.text[:500]}")
except httpx.TimeoutException:
    print(f"  TIMEOUT after 300s — HF Space may be down or overloaded")
except Exception as e:
    print(f"  Exception: {type(e).__name__}: {e}")

print()
print("Done.")
