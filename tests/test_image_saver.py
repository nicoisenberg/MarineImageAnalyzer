import numpy as np

from src.image_saver import print_save_status, save_image


def test_save_image(tmp_path):
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    output_path = tmp_path / "output_image.png"

    save_successful = save_image(image, str(output_path))

    assert save_successful
    assert output_path.exists()


def test_print_save_status_success(capsys):
    print_save_status(
        save_successful=True,
        image_label="Edge"
        )

    captured = capsys.readouterr()

    assert captured.out == "Edge image saved successfully.\n"


def test_print_status_failure(capsys):
    print_save_status(
        save_successful=False,
        image_label="Edge"
    )

    captured = capsys.readouterr()

    assert captured.out == "Failed to save edge image.\n"