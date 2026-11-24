import cv2


def _to_gray(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def global_threshold(image):
    # Global thresholding (Otsu): satu ambang untuk seluruh gambar
    gray = _to_gray(image)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, result = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return result


def adaptive_threshold(image):
    # Adaptive thresholding: ambang dihitung per area kecil (31x31 piksel)
    gray = _to_gray(image)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    result = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    return result
