import sys

from src.image_loader import load_image

image = load_image("images/sample_marine.jpg")

if image is None:
    sys.exit(1)

print("Image loaded successfully.")
print(f"Image shape: {image.shape}")
