import cv2


def resize_image(image, width, height):
    result = cv2.resize(image, (width, height))

    return result