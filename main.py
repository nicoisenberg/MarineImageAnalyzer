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

    return parser.parse_args()


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
            clip_limit=2.0,
            tile_grid_size=(8, 8)
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
        kernel_size=5
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
        lower_threshold=100,
        upper_threshold=200
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
        kernel_size=3
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
        min_area=100
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