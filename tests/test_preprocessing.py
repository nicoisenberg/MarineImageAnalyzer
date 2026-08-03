import numpy as np

from src.preprocessing import resize_image


def test_resize_image():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    resized_image = resize_image(
        image,
        width=50,
        height=25
    )

    assert resized_image.shape == (25, 50, 3)
    assert image.shape == (100, 200, 3)