import argparse

from pathlib import Path
from src.pipeline import process_image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyse marine images."
    )

    parser.add_argument(
        "--use-clahe",
        action="store_true",
        help="Apply CLAHE contrast enhancement before edge detection."
    )

    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        help="Clip limit of CLAHE."
    )

    parser.add_argument(
        "--clahe-grid",
        type=int,
        nargs=2,
        default=(8, 8),
        metavar=("X", "Y"),
        help="CLAHE grid size as two integers."
    )

    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=5,
        help="Kernel size for Gaussian blur."
    )

    parser.add_argument(
        "--canny-low",
        type=int,
        default=100,
        help="Lower threshold for Canny edge detection."
    )

    parser.add_argument(
        "--canny-high",
        type=int,
        default=200,
        help="Upper threshold for Canny edge detection."
    )

    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=3,
        help="Kernel size for morphological closing."
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum contour area in pixels."
    )

    args = parser.parse_args()

    if args.clahe_clip <= 0:
        parser.error("--clahe-clip must be a positive float.")

    if args.clahe_grid[0] <= 0 or args.clahe_grid[1] <= 0:
        parser.error("--clahe-grid values must both be positive integers.")

    if args.blur_kernel <= 0 or args.blur_kernel % 2 == 0:
        parser.error("--blur-kernel must be a positive odd integer.")

    if not 0 <= args.canny_low <= 255:
        parser.error("--canny-low must be between 0 and 255.")

    if not 0 <= args.canny_high <= 255:
        parser.error("--canny-high must be between 0 and 255.")
        
    if args.canny_low >= args.canny_high:
        parser.error("--canny-low must be lower than --canny-high.")

    if args.morph_kernel <= 0:
        parser.error("--morph-kernel must be a positive integer.")

    if args.min_area < 0:
        parser.error("--min-area must be 0 or larger.")

    return args


def main():
    args = parse_arguments()

    image_directory = Path("images")

    image_paths = image_directory.glob("*.jpg")

    for image_path in image_paths:
        output_directory = Path("output") / image_path.stem
        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )
        process_image(
            image_path,
            output_directory,
            args
        )
        

if __name__ == "__main__":
    main()