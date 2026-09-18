import cv2
import numpy as np


def resize_image(image, width, height):
    result = cv2.resize(image, (width, height))

    return result


def convert_to_grayscale(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return result


def apply_gaussian_blur(grayscale_image, kernel_size):
    result = cv2.GaussianBlur(grayscale_image, (kernel_size, kernel_size), 0)

    return result


def apply_morphological_closing(edge_image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    result = cv2.morphologyEx(
        edge_image,
        cv2.MORPH_CLOSE,
        kernel
        )

    return result