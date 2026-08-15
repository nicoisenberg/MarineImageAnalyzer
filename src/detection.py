import cv2


def detect_edges(grayscale_image, lower_threshold, upper_threshold):
    detected_edges = cv2.Canny(grayscale_image, lower_threshold, upper_threshold)

    return detected_edges


def find_contours(edge_image):
    contours, _ = cv2.findContours(
        edge_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def draw_contours(image, contours):
    result = image.copy()

    cv2.drawContours(
        result,
        contours,
        contourIdx=-1,
        color=(0, 0, 255),
        thickness=2
    )

    return result


def filter_contours_by_area(contours, min_area):
    relevant_contours = []

    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            relevant_contours.append(contour)

    return relevant_contours

