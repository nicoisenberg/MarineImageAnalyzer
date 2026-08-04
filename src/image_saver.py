import cv2


def save_image(image, output_path):
    save_successful = cv2.imwrite(output_path, image)

    return save_successful


def print_save_status(save_successful, image_label):
    if save_successful:
        print(f"{image_label} image saved successfully.")
    else:
        print(f"Failed to save {image_label.lower()} image.")

