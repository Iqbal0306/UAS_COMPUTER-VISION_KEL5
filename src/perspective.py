
import cv2
import numpy as np

def order_points(points):
    points = points.reshape(4, 2)

    rect = np.zeros((4, 2), dtype="float32")

    # Jumlah koordinat x + y
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # Top Left
    rect[2] = points[np.argmax(s)]  # Bottom Right

    # Selisih koordinat x - y
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  # Top Right
    rect[3] = points[np.argmax(diff)]  # Bottom Left

    return rect


def perspective_transform(image, points):
    rect = order_points(points)

    (tl, tr, br, bl) = rect

    # Hitung lebar hasil
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    # Hitung tinggi hasil
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    # Titik tujuan
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Matriks transformasi
    matrix = cv2.getPerspectiveTransform(rect, dst)

    # Transformasi perspektif
    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    return warped
