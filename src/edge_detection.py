
import cv2

def detect_edges(image):
    # Canny Edge Detection
    edges = cv2.Canny(
        image,
        threshold1=75,
        threshold2=200
    )

    # Memperkuat garis tepi
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (5, 5)
    )

    edges = cv2.dilate(
        edges, kernel, iterations=1
    )

    return edges
