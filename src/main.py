
import cv2
import matplotlib.pyplot as plt

from preprocessing import preprocess_image
from edge_detection import detect_edges
from contour_detection import find_document_contour
from perspective import perspective_transform
# Baca gambar
image = cv2.imread("../dataset/raw/document_1.jpeg")

if image is None:
    raise FileNotFoundError("Gambar tidak ditemukan!")

# Preprocessing
resized, gray, gaussian, median = preprocess_image(image)

# Edge detection
edges = detect_edges(gaussian)

# Contour detection
document = find_document_contour(edges)

if document is not None:
    result = resized.copy()

    # Gambar kontur dokumen
    cv2.drawContours(
        result, [document], -1,
        (0, 255, 0), 3
    )

    # Perspective Transform
    warped = perspective_transform(
        resized, document
    )

    # Simpan hasil
    cv2.imwrite(
        "../results/images/document_scan.jpg",
        warped
    )

    print("Dokumen berhasil dideteksi!")
    print("Perspective Transform berhasil!")
else:
    result = resized.copy()
    warped = None
    print("Dokumen belum berhasil dideteksi!")

# Visualisasi
images = [
    resized, gray, gaussian,
    median, edges, result
]

titles = [
    "Original", "Grayscale",
    "Gaussian Blur", "Median Blur",
    "Canny Edge", "Contour Detection"
]

plt.figure(figsize=(14, 9))

for i in range(6):
    plt.subplot(2, 3, i + 1)

    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(
            images[i], cv2.COLOR_BGR2RGB
        ))

    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

# Tampilkan hasil scan
if warped is not None:
    plt.figure(figsize=(7, 9))
    plt.imshow(cv2.cvtColor(
        warped, cv2.COLOR_BGR2RGB
    ))
    plt.title("Perspective Transform Result")
    plt.axis("off")
    plt.show()

