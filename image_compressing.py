from PIL import Image
import os

# Root directory to search for images
ROOT_DIR = "./public"  # Update this if your path is different

# JPEG quality setting (lower = more compression)
JPEG_QUALITY = 75

# Walk through all subdirectories
for folder, _, files in os.walk(ROOT_DIR):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(folder, filename)

            try:
                img = Image.open(file_path)

                # For JPEGs: Convert to RGB if needed and compress
                if filename.lower().endswith((".jpg", ".jpeg")):
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.save(file_path, "JPEG", quality=JPEG_QUALITY, optimize=True)

                # For PNGs: Just optimize
                elif filename.lower().endswith(".png"):
                    img.save(file_path, "PNG", optimize=True)

                print(f"✅ Compressed: {file_path}")

            except Exception as e:
                print(f"❌ Failed to compress {file_path}: {e}")