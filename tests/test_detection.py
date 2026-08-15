import numpy as np

from src.detection import detect_edges, draw_contours, find_contours


def test_detect_edges():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[25:75, 25:75] = 255

    edge_image = detect_edges(
        image,
        lower_threshold=100,
        upper_threshold=200
    )

    assert edge_image.shape == (100, 100)
    assert np.any(edge_image > 0)


def test_find_contours():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[25:75, 25:75] = 255

    contours = find_contours(image)

    assert len(contours) == 1


def test_draw_contours():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255

    contours = find_contours(mask)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    original_image = image.copy()

    image_contours = draw_contours(
        image,
        contours
    )

    assert image_contours.shape == image.shape
    assert np.array_equal(image, original_image)

    red_pixels = np.all(
        image_contours == [0, 0, 255],
        axis=2
    )

    assert np.any(red_pixels)