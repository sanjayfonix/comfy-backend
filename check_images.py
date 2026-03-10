import os, glob
from PIL import Image

print("=== INPUT IMAGES ===")
p = Image.open('test_images/test_person.jpg')
print(f"Person:  {p.size} {p.mode}, {os.path.getsize('test_images/test_person.jpg')/1024:.0f}KB")

g_path = r"C:\Users\Admin\Downloads\ComifyAiProject-main\ComifyAiProject-main\public\JColeFront.webp"
g = Image.open(g_path)
print(f"Garment: {g.size} {g.mode}, {os.path.getsize(g_path)/1024:.0f}KB")

print("\n=== HF SPACE RESULT ===")
r = Image.open('outputs/hf_test_result.jpg')
print(f"Result:  {r.size} {r.mode}, {os.path.getsize('outputs/hf_test_result.jpg')/1024:.0f}KB")

print("\n=== RECENT LOCAL COMPOSITES ===")
outputs = sorted(glob.glob('outputs/tryon_*.jpg'), key=os.path.getmtime, reverse=True)[:3]
for f in outputs:
    img = Image.open(f)
    print(f"  {os.path.basename(f)}: {img.size} ({os.path.getsize(f)/1024:.0f}KB)")
