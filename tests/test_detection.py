import numpy as np

from src.detection import detect_edges


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

