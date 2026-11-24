
import cv2

def preprocess_image(image):
    # Resize gambar
    height, width = image.shape[:2]
    scale = 800 / width

    resized = cv2.resize(
        image,
        (800, int(height * scale))
    )

    # Konversi ke grayscale
    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    # Gaussian Blur
    gaussian = cv2.GaussianBlur(
        gray, (5, 5), 0
    )

    # Median Blur
    median = cv2.medianBlur(
        gray, 5
    )

    return resized, gray, gaussian, median
