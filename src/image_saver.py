import cv2


def save_image(image, output_path):
    save_successful = cv2.imwrite(output_path, image)

    return save_successful
