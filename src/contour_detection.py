import cv2


def find_document_contour(edges, min_area_ratio=None, require_convex=False):
    # min_area_ratio None  : luas minimal 1000 piksel (perilaku awal)
    # min_area_ratio 0.2   : luas minimal 20% dari luas gambar
    # require_convex True  : hanya menerima bentuk cembung
    if min_area_ratio is None:
        min_area = 1000
    else:
        min_area = min_area_ratio * edges.shape[0] * edges.shape[1]

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
        if cv2.contourArea(contour) <= min_area:
            break

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
            if require_convex and not cv2.isContourConvex(approximation):
                continue
            return approximation

    return None
