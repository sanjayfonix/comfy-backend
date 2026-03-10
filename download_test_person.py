"""Download a proper high-resolution test person image from the IDM-VTON examples."""
import httpx
from pathlib import Path

# IDM-VTON example person images from their HuggingFace Space
URLS = [
    # Example person images from the yisol/IDM-VTON space
    "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/example/human/00008_00.jpg",
    "https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/example/human/00034_00.jpg",
]

out_dir = Path("test_images")
out_dir.mkdir(exist_ok=True)

for i, url in enumerate(URLS):
    fname = url.split("/")[-1]
    dest = out_dir / fname
    print(f"Downloading {fname}...")
    try:
        r = httpx.get(url, follow_redirects=True, timeout=30)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            print(f"  Saved {dest} ({len(r.content) / 1024:.0f} KB)")
        else:
            print(f"  HTTP {r.status_code}")
    except Exception as e:
        print(f"  Error: {e}")

print("\nDone. Use these for better test results!")
