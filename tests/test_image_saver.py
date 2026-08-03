import numpy as np

from src.image_saver import save_image


def test_save_image(tmp_path):
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    output_path = tmp_path / "output_image.png"

    save_successful = save_image(image, str(output_path))

    assert save_successful
    assert output_path.exists()