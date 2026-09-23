import argparse
import sys

from src.analysis import calculate_contour_statistics
from src.detection import detect_edges, draw_bounding_boxes, draw_contours, filter_contours_by_area, find_contours
from src.image_loader import load_image
from src.image_saver import print_save_status, save_image
from src.preprocessing import apply_clahe, apply_gaussian_blur, apply_morphological_closing, convert_to_grayscale, resize_image


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

    image = load_image("images/sample_marine.jpg")

    if image is None:
        sys.exit(1)

    print("Image loaded successfully.")
    print(f"Original image shape: {image.shape}")

    # Standardize the image for subsequent processing
    resized_image = resize_image(
        image,
        width=800,
        height=600
    )

    print("Image resized successfully.")
    print(f"Resized image shape: {resized_image.shape}")

    save_successful = save_image(
        resized_image,
        "output/resized_image.jpg"
    )

    print_save_status(
        save_successful,
        image_label="Resized"
    )

    grayscale_image = convert_to_grayscale(resized_image)

    print("Image converted to grayscale successfully.")
    print(f"Grayscale image shape: {grayscale_image.shape}")

    save_successful = save_image(
        grayscale_image,
        "output/grayscale_image.jpg"
    )

    print_save_status(
        save_successful,
        image_label="Grayscale"
    )

    preprocessed_image = grayscale_image

    if args.use_clahe:

        enhanced_image = apply_clahe(
            grayscale_image=grayscale_image,
            clip_limit=args.clahe_clip,
            tile_grid_size=tuple(args.clahe_grid)
        )

        print("Image contrast enhanced successfully.")
        print(f"Enhanced image shape: {enhanced_image.shape}")

        save_successful = save_image(
            enhanced_image,
            "output/enhanced_image.jpg"
        )

        print_save_status(
            save_successful,
            image_label="Enhanced"
        )

        preprocessed_image = enhanced_image

    # Reduce image noise before edge detection
    blurred_image = apply_gaussian_blur(
        preprocessed_image,
        kernel_size=args.blur_kernel
    )

    print("Gaussian blur applied successfully.")
    print(f"Blurred image shape: {blurred_image.shape}")

    save_successful = save_image(
        blurred_image,
        "output/blurred_image.jpg"
    )

    print_save_status(
        save_successful,
        image_label="Blurred"
    )

    edge_image = detect_edges(
        blurred_image,
        lower_threshold=args.canny_low,
        upper_threshold=args.canny_high
    )

    print("Image edges detected successfully.")
    print(f"Edge image shape: {edge_image.shape}")

    save_successful = save_image(
        edge_image,
        "output/edge_image.jpg"
    )

    print_save_status(
        save_successful,
        image_label="Edge"
    )

    closed_edge_image = apply_morphological_closing(
        edge_image=edge_image,
        kernel_size=args.morph_kernel
    )

    print("Image edges were closed successfully.")
    print(f"Closed edge image shape: {closed_edge_image.shape}")

    save_successful = save_image(
        closed_edge_image,
        "output/closed_edge_image.jpg"
    )

    print_save_status(
        save_successful,
        image_label="Closed edge"
    )

    original_contours = find_contours(edge_image)
    closed_contours = find_contours(closed_edge_image)

    print(f"Detected original contours: {len(original_contours)}")
    print(f"Detected closed contours: {len(closed_contours)}")

    image_contours = draw_contours(
        resized_image,
        closed_contours
    )

    print("Contours drawn on image successfully.")
    print(f"Image with contours shape: {image_contours.shape}")

    save_successful = save_image(
        image_contours,
        "output/image_contours.jpg"
    )

    print_save_status(
        save_successful,
        image_label="Contours"
    )

    relevant_contours = filter_contours_by_area(
        closed_contours,
        min_area=args.min_area
    )

    print(f"Detected relevant contours: {len(relevant_contours)}")

    bounding_box_image = draw_bounding_boxes(
        image=resized_image,
        contours=relevant_contours
    )

    print("Boxes drawn on image successfully.")
    print(f"Bounding box image shape: {bounding_box_image.shape}")

    save_successful = save_image(
        bounding_box_image,
        "output/bounding_box_image.jpg"
    )

    print_save_status(
        save_successful=save_successful,
        image_label="Bounding box"
    )

    original_contour_stats = calculate_contour_statistics(original_contours)
    print(
        "\nOriginal contour statistics:\n"
        f"Count: {original_contour_stats['count']}\n"
        f"Total Area: {original_contour_stats['total_area']}\n"
        f"Average Area: {original_contour_stats['average_area']}\n"
        f"Largest Area: {original_contour_stats['largest_area']}"
    )

    closed_contour_stats = calculate_contour_statistics(closed_contours)
    print(
        "\nClosed contour statistics:\n"
        f"Count: {closed_contour_stats['count']}\n"
        f"Total Area: {closed_contour_stats['total_area']}\n"
        f"Average Area: {closed_contour_stats['average_area']}\n"
        f"Largest Area: {closed_contour_stats['largest_area']}"
    )

    relevant_contour_stats = calculate_contour_statistics(relevant_contours)
    print(
        "\nRelevant contour statistics:\n"
        f"Count: {relevant_contour_stats['count']}\n"
        f"Total Area: {relevant_contour_stats['total_area']}\n"
        f"Average Area: {relevant_contour_stats['average_area']}\n"
        f"Largest Area: {relevant_contour_stats['largest_area']}"
    )


if __name__ == "__main__":
    main()