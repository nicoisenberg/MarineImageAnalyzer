import cv2


def calculate_contour_statistics(contours):
    areas = []

    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)

    if not areas:
        count = 0
        total_area = 0
        average_area = 0
        largest_area = 0
    else:
        count = len(areas)
        total_area = sum(areas)
        average_area = total_area / count
        largest_area = max(areas)

    statistics = {
        "count": count,
        "total_area": total_area,
        "average_area": average_area,
        "largest_area": largest_area
    }

    return statistics