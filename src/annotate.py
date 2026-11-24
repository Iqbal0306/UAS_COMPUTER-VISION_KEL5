"""
Alat bantu anotasi: klik 4 sudut dokumen pada tiap foto untuk membuat ground truth.

    python src/annotate.py --condition normal
    python src/annotate.py --condition miring
    python src/annotate.py --condition redup

Kontrol (di jendela gambar):
    klik kiri  = tandai sudut (urutan bebas, 4 titik)
    z          = batalkan titik terakhir
    n / Enter  = simpan dan lanjut ke foto berikutnya
    s          = lewati foto ini
    q          = keluar

Hasil ditambahkan ke dataset/annotations/corners.csv
Foto yang sudah punya anotasi otomatis dilewati.
"""
import argparse
import csv
import os
from pathlib import Path

import cv2

EXT = (".jpg", ".jpeg", ".png", ".bmp")
SHOW_H = 900


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset/raw")
    ap.add_argument("--csv", default="dataset/annotations/corners.csv")
    ap.add_argument("--condition", default="normal",
                    help="label kondisi foto: normal / miring / redup")
    ap.add_argument("--prefix", default="",
                    help="hanya foto yang namanya diawali teks ini")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    done = set()
    new_file = not os.path.exists(args.csv)
    if not new_file:
        with open(args.csv, newline="", encoding="utf-8") as f:
            done = {r["filename"] for r in csv.DictReader(f)}

    files = [p for p in sorted(Path(args.input).iterdir())
             if p.suffix.lower() in EXT and p.name.startswith(args.prefix)
             and p.name not in done]

    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["filename", "condition", "x1", "y1", "x2", "y2",
                             "x3", "y3", "x4", "y4"])
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            scale = SHOW_H / img.shape[0]
            view = cv2.resize(img, None, fx=scale, fy=scale)
            pts = []

            def on_click(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                    pts.append((x, y))

            win = "Anotasi - " + p.name
            cv2.namedWindow(win)
            cv2.setMouseCallback(win, on_click)
            action = None
            while True:
                canvas = view.copy()
                for i, (x, y) in enumerate(pts):
                    cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
                    cv2.putText(canvas, str(i + 1), (x + 8, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.imshow(win, canvas)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("z") and pts:
                    pts.pop()
                elif key in (ord("n"), 13) and len(pts) == 4:
                    action = "save"
                    break
                elif key == ord("s"):
                    action = "skip"
                    break
                elif key == ord("q"):
                    action = "quit"
                    break
            cv2.destroyAllWindows()

            if action == "save":
                row = [p.name, args.condition]
                for x, y in pts:
                    row += [round(x / scale, 1), round(y / scale, 1)]
                writer.writerow(row)
                f.flush()
                print("tersimpan:", p.name)
            elif action == "quit":
                break


if __name__ == "__main__":
    main()
