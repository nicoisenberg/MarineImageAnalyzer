import sys

from src.detection import detect_edges, draw_contours, find_contours
from src.image_loader import load_image
from src.image_saver import print_save_status, save_image
from src.preprocessing import apply_gaussian_blur, convert_to_grayscale, resize_image


image = load_image("images/sample_marine.jpg")

#Load image
if image is None:
    sys.exit(1)

print("Image loaded successfully.")
print(f"Original image shape: {image.shape}")

# Standardize the image for subsequent processing
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

print_save_status(
    save_successful,
    image_label="Resized"
)

grayscale_image = convert_to_grayscale(resized_image)

print("Image converted to grayscale successfully.")
print(f"Grayscale image shape: {grayscale_image.shape}")

save_successful = save_image(
    grayscale_image,
    "output/grayscale_image.jpg"
)

print_save_status(
    save_successful,
    image_label="Grayscale"
)

# Reduce image noise before edge detection
blurred_image = apply_gaussian_blur(
    grayscale_image,
    kernel_size=5
)

print("Gaussian blur applied successfully.")
print(f"Blurred image shape: {blurred_image.shape}")

save_successful = save_image(
    blurred_image,
    "output/blurred_image.jpg"
)

print_save_status(
    save_successful,
    image_label="Blurred"
)

edge_image = detect_edges(
    blurred_image,
    lower_threshold=100,
    upper_threshold=200
)

print("Image edges detected successfully.")
print(f"Edge image shape: {edge_image.shape}")

save_successful = save_image(
    edge_image,
    "output/edge_image.jpg"
)

print_save_status(
    save_successful,
    image_label="Edge"
)

contours = find_contours(edge_image)

print(f"Detected contours: {len(contours)}")

image_contours = draw_contours(
    resized_image,
    contours
)

print("Contours drawn on image successfully.")
print(f"Image with contours shape: {image_contours.shape}")

save_successful = save_image(
    image_contours,
    "output/image_contours.jpg"
)

print_save_status(
    save_successful,
    image_label="Contours"
)