import numpy as np

from src.preprocessing import apply_gaussian_blur, convert_to_grayscale, resize_image 


def test_resize_image():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    resized_image = resize_image(
        image,
        width=50,
        height=25
    )

    assert resized_image.shape == (25, 50, 3)
    assert image.shape == (100, 200, 3)


def test_grayscale_conversion():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    grayscale_image = convert_to_grayscale(image)

    assert grayscale_image is not None
    assert grayscale_image.shape == (100, 200)


def test_gaussian_blur():
    image = np.zeros((100, 200), dtype=np.uint8)
    image[50, 100] = 255

    blurred_image = apply_gaussian_blur(
        image,
        kernel_size=5
    )

    assert blurred_image is not None
    assert blurred_image.shape == image.shape
    assert blurred_image[50, 100] < 255
    assert np.any(blurred_image > 0)
