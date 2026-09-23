import sys

import pytest

from main import parse_arguments


def test_valid_cli(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
        "main.py",
        "--use-clahe",
        "--clahe-clip", "3",
        "--clahe-grid", "4", "6",
        "--blur-kernel", "7",
        "--canny-low", "50",
        "--canny-high", "150",
        "--morph-kernel", "5",
        "--min-area", "300"
        ]
    )

    args = parse_arguments()

    assert args.use_clahe
    assert args.clahe_clip == 3.0
    assert args.clahe_grid == [4, 6]
    assert args.blur_kernel == 7
    assert args.canny_low == 50
    assert args.canny_high == 150
    assert args.morph_kernel == 5
    assert args.min_area == 300


def test_invalid_clahe_clip(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--clahe-clip", "0"]
    )

    with pytest.raises(SystemExit):
        parse_arguments()


def test_invalid_clahe_grid(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--clahe-grid", "0", "4"]
    )

    with pytest.raises(SystemExit):
        parse_arguments()


def test_invalid_blur_kernel(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--blur-kernel", "4"]
    )

    with pytest.raises(SystemExit):
        parse_arguments()


def test_invalid_canny_low(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--canny-low", "-1"]
    )

    with pytest.raises(SystemExit):
        parse_arguments()


def test_invalid_canny_high(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--canny-high", "300"]
    )

    with pytest.raises(SystemExit):
        parse_arguments()


def test_invalid_canny_threshold_order(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--canny-low", "200",
            "--canny-high", "100"
        ]
    )

    with pytest.raises(SystemExit):
        parse_arguments()


def test_invalid_morph_kernel(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--morph-kernel", "0"]
    )

    with pytest.raises(SystemExit):
        parse_arguments()


def test_invalid_min_area(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--min-area", "-5"]
    )

    with pytest.raises(SystemExit):
        parse_arguments()