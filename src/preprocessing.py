import cv2


def resize_image(image, width, height):
    result = cv2.resize(image, (width, height))

    return result


def convert_to_grayscale(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return result

