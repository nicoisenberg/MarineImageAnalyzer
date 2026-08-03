import cv2

def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image: {image_path}")
        return None

    return image