import cv2
import numpy as np

from src.image_loader import load_image

def test_load_existing_image(tmp_path):
    image_path = tmp_path / "test_image.png"

    test_image = np.zeros((10, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), test_image)

    loaded_image = load_image(str(image_path))

    assert loaded_image is not None
    assert loaded_image.shape == (10, 20, 3)

def test_load_missing_image(tmp_path):
    image_path = tmp_path / "missing_image.png"

    loaded_image = load_image(str(image_path))

    assert loaded_image is None