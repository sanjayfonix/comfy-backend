"""Test HF Space call through the actual async endpoint handler."""
import os, sys, asyncio, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from main import (
    generate_tryon_base64, Base64TryOnRequest,
    image_to_base64, Image, call_hf_space_tryon,
    USE_HF_SPACE, HF_SPACE_ID, HF_TOKEN,
)


async def test():
    person_img = Image.open('test_images/test_person.jpg').convert('RGB')
    garment_img = Image.open('test_images/test_garment.jpg').convert('RGB')

    person_b64 = image_to_base64(person_img)
    garment_b64 = image_to_base64(garment_img)

    print(f"USE_HF_SPACE={USE_HF_SPACE}")
    print(f"HF_SPACE_ID={HF_SPACE_ID}")
    print(f"HF_TOKEN={HF_TOKEN[:15]}...")

    # Test 1: Direct call to HF Space
    print("\n--- Test 1: Direct HF Space call ---")
    try:
        t0 = time.time()
        result_img = await call_hf_space_tryon(person_img, garment_img, category='upper_body')
        print(f"SUCCESS in {time.time()-t0:.1f}s, size={result_img.size}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

    # Test 2: Through the endpoint handler
    print("\n--- Test 2: Through endpoint handler ---")
    req = Base64TryOnRequest(person_image=person_b64, garment_image=garment_b64, category='upper_body')
    t0 = time.time()
    result = await generate_tryon_base64(req)
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s")
    has_model = 'model' in result
    print(f"Has model key: {has_model}")
    if has_model:
        print(f"Model: {result['model']}")
    else:
        print("MODEL MISSING - used local composite fallback!")


asyncio.run(test())
