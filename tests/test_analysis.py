import numpy as np

from src.analysis import calculate_contour_statistics


def test_calculate_contour_statistics_existing():

    contour_1 = np.array(
        [[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]],
        dtype=np.int32
    )

    contour_2 = np.array(
        [[[0, 0]], [[20, 0]], [[20, 10]], [[0, 10]]],
        dtype=np.int32
    )

    contour_3 = np.array(
        [[[0, 0]], [[30, 0]], [[30, 20]], [[0, 20]]],
        dtype=np.int32
    )

    contours = [contour_1, contour_2, contour_3]

    contour_stats = calculate_contour_statistics(contours)

    validation_contour_stats = {
        "count": 3,
        "total_area": 900,
        "average_area": 300,
        "largest_area": 600
    }

    assert validation_contour_stats == contour_stats


def test_calculate_contour_statistics_empty():
    contours = []
    contour_stats = calculate_contour_statistics(contours)

    validation_contour_stats = {
            "count": 0,
            "total_area": 0,
            "average_area": 0,
            "largest_area": 0
        }

    assert validation_contour_stats == contour_stats