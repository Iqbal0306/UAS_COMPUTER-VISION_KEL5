"""
Entry point P01 - Automatic Document Scanner

    python src/main.py                         # evaluasi semua foto di dataset/raw
    python src/main.py --canny auto            # evaluasi dengan ambang Canny otomatis
    python src/main.py --image dataset/raw/normal_01.jpg --show
    python src/main.py --demo                  # buat demo/demo.mp4

Opsi lain untuk mode evaluasi: lihat  python src/evaluate.py --help
"""
import argparse
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import preprocess_image
from edge_detection import detect_edges
from contour_detection import find_document_contour
from perspective import perspective_transform, order_points
from enhancement import global_threshold, adaptive_threshold

ROOT = Path(__file__).resolve().parent.parent
EXT = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXT)


def run_pipeline(image, filter_name="gaussian", canny="fixed",
                 min_area_ratio=None, convex=False):
    resized, gray, gaussian, median = preprocess_image(image)
    filtered = gaussian if filter_name == "gaussian" else median
    edges = detect_edges(filtered, canny)
    contour = find_document_contour(edges, min_area_ratio, convex)

    marked = resized.copy()
    warped = otsu = adaptive = None
    if contour is not None:
        cv2.drawContours(marked, [contour], -1, (0, 255, 0), 3)
        for (x, y) in order_points(contour).astype(int):
            cv2.circle(marked, (int(x), int(y)), 8, (0, 0, 255), -1)
        warped = perspective_transform(resized, contour)
        otsu = global_threshold(warped)
        adaptive = adaptive_threshold(warped)

    return {"resized": resized, "gray": gray, "filtered": filtered,
            "edges": edges, "marked": marked, "warped": warped,
            "otsu": otsu, "adaptive": adaptive, "found": contour is not None}


def run_single(path, args):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")

    r = run_pipeline(image, args.filter, args.canny)
    print("Dokumen terdeteksi." if r["found"] else "Dokumen belum terdeteksi.")

    out_dir = ROOT / "results" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [("Asli", r["resized"]), ("Grayscale", r["gray"]),
              (f"Filter {args.filter}", r["filtered"]), ("Canny + dilasi", r["edges"]),
              ("Deteksi sudut", r["marked"]), ("Perspective", r["warped"]),
              ("Global (Otsu)", r["otsu"]), ("Adaptive", r["adaptive"])]

    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (title, im) in zip(axs.ravel(), panels):
        ax.set_title(title)
        ax.axis("off")
        if im is None:
            continue
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    save = out_dir / f"{Path(path).stem}_pipeline.png"
    fig.savefig(save, dpi=100)
    print("Tersimpan:", save)
    if args.show:
        plt.show()
    plt.close(fig)

    if r["found"]:
        img_dir = ROOT / "results" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        cv2.imwrite(str(img_dir / f"{stem}_scan.png"), r["warped"])
        cv2.imwrite(str(img_dir / f"{stem}_scan_global_otsu.png"), r["otsu"])
        cv2.imwrite(str(img_dir / f"{stem}_scan_adaptive.png"), r["adaptive"])


# ------------------------------------------------------------------ demo video
W, H, FPS = 1280, 720, 24


def make_frame(image, caption, sub=""):
    canvas = np.full((H, W, 3), 25, np.uint8)
    if image is not None:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        area_h = H - 120
        scale = min(W / image.shape[1], area_h / image.shape[0])
        nw, nh = int(image.shape[1] * scale), int(image.shape[0] * scale)
        small = cv2.resize(image, (nw, nh))
        x0, y0 = (W - nw) // 2, 100 + (area_h - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = small
    cv2.putText(canvas, caption, (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (255, 255, 255), 3, cv2.LINE_AA)
    if sub:
        cv2.putText(canvas, sub, (30, H - 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (180, 180, 180), 2, cv2.LINE_AA)
    return canvas


def make_demo_video(args):
    files = list_images(ROOT / "dataset" / "raw")[:args.max_images]
    if not files:
        print("Tidak ada foto di dataset/raw")
        return
    out_path = ROOT / "demo" / "demo.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter gagal dibuka (codec mp4v tidak tersedia)")

    hold = int(FPS * args.seconds)
    for p in files:
        image = cv2.imread(str(p))
        if image is None:
            continue
        r = run_pipeline(image, args.filter, args.canny)
        stages = [
            (r["resized"], "1. Foto asli"),
            (r["gray"], "2. Grayscale"),
            (r["filtered"], f"3. Filter {args.filter}"),
            (r["edges"], "4. Canny edge detection + dilasi"),
        ]
        if r["found"]:
            stages += [
                (r["marked"], "5. Deteksi 4 sudut dokumen"),
                (r["warped"], "6. Perspective transform"),
                (r["otsu"], "7. Global thresholding (Otsu)"),
                (r["adaptive"], "8. Adaptive thresholding"),
            ]
        else:
            stages.append((r["marked"], "5. Dokumen TIDAK terdeteksi"))
        for im, cap in stages:
            frame = make_frame(im, cap, p.name)
            for _ in range(hold):
                writer.write(frame)
    writer.release()
    print("Video tersimpan:", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="proses satu foto")
    ap.add_argument("--show", action="store_true", help="tampilkan jendela grafik")
    ap.add_argument("--demo", action="store_true", help="buat demo/demo.mp4")
    ap.add_argument("--filter", choices=["gaussian", "median"], default="gaussian")
    ap.add_argument("--canny", choices=["fixed", "auto"], default="fixed")
    ap.add_argument("--max-images", type=int, default=6, help="jumlah foto di video demo")
    ap.add_argument("--seconds", type=float, default=1.2, help="durasi tiap tahap (detik)")
    args, _ = ap.parse_known_args()

    if args.image:
        run_single(args.image, args)
    elif args.demo:
        make_demo_video(args)
    else:
        import evaluate
        evaluate.main(sys.argv[1:])


if __name__ == "__main__":
    main()
