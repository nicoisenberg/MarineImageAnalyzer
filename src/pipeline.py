from src.analysis import calculate_contour_statistics
from src.detection import detect_edges, draw_bounding_boxes, draw_contours, filter_contours_by_area, find_contours
from src.image_loader import load_image
from src.image_saver import print_save_status, save_image
from src.preprocessing import apply_clahe, apply_gaussian_blur, apply_morphological_closing, convert_to_grayscale, resize_image


def process_image(image_path, output_directory, args):
    print(f"\nProcessing image: {image_path.name}")
    
    image = load_image(image_path)
    
    if image is None:
        return

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
        output_directory / "resized_image.jpg"
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
        output_directory / "grayscale_image.jpg"
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
            output_directory / "enhanced_image.jpg"
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
        output_directory / "blurred_image.jpg"
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
        output_directory / "edge_image.jpg"
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
        output_directory / "closed_edge_image.jpg"
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
        output_directory / "image_contours.jpg"
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
        output_directory / "bounding_box_image.jpg"
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

    return {
        "image": image_path.name,
        "original_contours": original_contour_stats["count"],
        "closed_contours": closed_contour_stats["count"],
        "relevant_contours": relevant_contour_stats["count"],
        "relevant_total_area": relevant_contour_stats["total_area"],
        "relevant_average_area": relevant_contour_stats["average_area"],
        "relevant_largest_area": relevant_contour_stats["largest_area"]
    }