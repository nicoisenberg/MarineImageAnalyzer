import cv2


def detect_edges(grayscale_image, lower_threshold, upper_threshold):
    detected_edges = cv2.Canny(grayscale_image, lower_threshold, upper_threshold)

    return detected_edges