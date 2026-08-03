import sys

from src.image_loader import load_image
from src.image_saver import save_image
from src.preprocessing import resize_image


image = load_image("images/sample_marine.jpg")

if image is None:
    sys.exit(1)

print("Image loaded successfully.")
print(f"Original image shape: {image.shape}")

resized_image = resize_image(
    image,
    width=800,
    height=600
)

print("Image resized successfully.")
print(f"Resized image shape: {resized_image.shape}")

save_successful = save_image(
    resized_image,
    "output/resized_image.jpg"
)

if save_successful:
    print("Resized image saved successfully.")
else:
    print("Failed to save resized image.")