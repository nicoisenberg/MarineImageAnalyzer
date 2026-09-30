import csv

def save_summary(results, output_path):

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)