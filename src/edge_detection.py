import cv2
import numpy as np


def detect_edges(image, mode="fixed"):
    # mode "fixed": ambang Canny tetap (75, 200)
    # mode "auto" : ambang dihitung dari median intensitas gambar
    if mode == "auto":
        median = float(np.median(image))
        low = int(max(0, 0.66 * median))
        high = int(min(255, 1.33 * median))
    else:
        low, high = 75, 200

    edges = cv2.Canny(
        image,
        threshold1=low,
        threshold2=high
    )

    # Memperkuat garis tepi
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (5, 5)
    )

    edges = cv2.dilate(
        edges, kernel, iterations=1
    )

    return edges
