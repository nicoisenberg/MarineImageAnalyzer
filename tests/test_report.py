import csv

from src.report import save_summary

def test_save_summary(tmp_path):
    results = [
        {
            "image": "shark.jpg",
            "original_contours": 10,
            "closed_contours": 8,
            "relevant_contours": 4,
            "relevant_total_area": 500.0,
            "relevant_average_area": 125.0,
            "relevant_largest_area": 200.0
        },
        {
            "image": "turtle.jpg",
            "original_contours": 20,
            "closed_contours": 15,
            "relevant_contours": 5,
            "relevant_total_area": 1000.0,
            "relevant_average_area": 200.0,
            "relevant_largest_area": 400.0
        }
    ]

    output_path = tmp_path / "summary.csv"

    save_summary(
        results,
        output_path
    )

    assert output_path.exists()

    with open(output_path, "r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["image"] == "shark.jpg"
    assert rows[0]["relevant_contours"] == "4"
    assert rows[1]["image"] == "turtle.jpg"
    assert rows[1]["relevant_largest_area"] == "400.0"