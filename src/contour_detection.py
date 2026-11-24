
import cv2

def find_document_contour(edges):
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Urutkan berdasarkan luas terbesar
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours:
        perimeter = cv2.arcLength(
            contour, True
        )

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        # Cari kontur dengan empat sudut
        if len(approximation) == 4:
            if cv2.contourArea(contour) > 1000:
                return approximation

    return None
