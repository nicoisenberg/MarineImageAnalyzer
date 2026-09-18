import numpy as np

from src.preprocessing import apply_clahe, apply_gaussian_blur, apply_morphological_closing, convert_to_grayscale, resize_image 


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


def test_morphological_closing():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[25:50, 20] = 255
    image[51:60, 20] = 255

    closed_edge_image = apply_morphological_closing(
        edge_image=image,
        kernel_size=3
    )

    assert image.shape == closed_edge_image.shape
    assert closed_edge_image[50, 20] == 255


def test_apply_clahe():
    image = np.full((100, 100), 100, dtype=np.uint8)
    image[20:30, 20:40] = 110

    enhanced_image = apply_clahe(
        grayscale_image=image,
        clip_limit=2.0,
        tile_grid_size=(8, 8)
    )

    assert image.shape == enhanced_image.shape
    assert image.dtype == enhanced_image.dtype
    assert not np.array_equal(image, enhanced_image)