"""Quick test script for the backend APIs."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from main import detect_pose, pil_from_upload, recommend_size, estimate_measurements_from_landmarks, composite_tryon
from PIL import Image
import traceback

TEST_IMG = os.path.join(os.path.dirname(__file__), "test_images", "test_person.jpg")

print("=" * 50)
print("Testing Size Recommendation Pipeline")
print("=" * 50)

try:
    img = Image.open(TEST_IMG).convert("RGB")
    print(f"  Loaded image: {img.size}")
    
    result = detect_pose(img)
    print(f"  Body Detected: {result['bodyDetected']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Recommended Size: {result['recommendedSize']}")
    print(f"  Measurements:")
    for k, v in result['bodyMeasurements'].items():
        print(f"    {k}: {v} cm")
    print(f"  Fit Prediction: {result['clothingFitPrediction']}")
    print(f"  Landmarks: {len(result['poseDetection']['landmarks'])} detected")
    print("\n  SIZE RECOMMENDATION: WORKING!")
except Exception as e:
    print(f"\n  SIZE RECOMMENDATION FAILED:")
    traceback.print_exc()

print("\n" + "=" * 50)
print("Testing Virtual Try-On Pipeline")
print("=" * 50)

GARMENT_IMG = os.path.join(os.path.dirname(__file__), "test_images", "test_garment.jpg")

try:
    person_img = Image.open(TEST_IMG).convert("RGB")
    garment_img = Image.open(GARMENT_IMG).convert("RGB")
    print(f"  Loaded person: {person_img.size}")
    print(f"  Loaded garment: {garment_img.size}")
    
    pose_data = detect_pose(person_img)
    landmarks = pose_data['poseDetection']['landmarks']
    print(f"  Pose detected, {len(landmarks)} landmarks")
    
    result_img = composite_tryon(person_img, garment_img, landmarks)
    output_path = os.path.join(os.path.dirname(__file__), "test_images", "test_output.jpg")
    result_img.save(output_path, "JPEG", quality=90)
    print(f"  Try-on image generated: {result_img.size}")
    print(f"  Saved to: {output_path}")
    print("\n  VIRTUAL TRY-ON: WORKING!")
except Exception as e:
    print(f"\n  VIRTUAL TRY-ON FAILED:")
    traceback.print_exc()
