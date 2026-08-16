import numpy as np

from src.detection import detect_edges, draw_bounding_boxes, draw_contours, filter_contours_by_area, find_contours


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


def test_filter_contour_by_area():
    image_small_rectangle = np.zeros((100, 100), dtype=np.uint8)
    image_small_rectangle[25:30, 25:30] = 255
    image_large_rectangle = np.zeros((100, 100), dtype=np.uint8)
    image_large_rectangle[25:75, 25:75] = 255

    contours_small_rectangle = find_contours(image_small_rectangle)
    contours_large_rectangle = find_contours(image_large_rectangle)

    relevant_contours_small_rectangle = filter_contours_by_area(
        contours_small_rectangle,
        min_area=500
    )

    relevant_contours_large_rectangle = filter_contours_by_area(
        contours_large_rectangle,
        min_area=500
    )

    assert len(relevant_contours_small_rectangle) == 0
    assert len(relevant_contours_large_rectangle) == 1


def test_draw_bounding_boxes():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255

    contours = find_contours(mask)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    original_image = image.copy()

    bounding_box_image = draw_bounding_boxes(
        image=image,
        contours=contours
    )

    assert bounding_box_image.shape == image.shape
    assert np.array_equal(image, original_image)

    red_pixels = np.all(
        bounding_box_image == [0, 0, 255],
        axis=2
    )

    assert np.any(red_pixels)