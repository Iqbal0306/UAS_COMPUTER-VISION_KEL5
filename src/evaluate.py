"""
Evaluasi eksperimen P01 - memakai modul buatan sendiri.

Jalankan dari folder mana saja:
    python src/evaluate.py
    python src/evaluate.py --tol 20

Eksperimen 1: Gaussian vs Median (sebelum Canny)
Eksperimen 2: Global (Otsu) vs Adaptive thresholding (setelah transform)
Metrik: detection rate, correct rate, corner error (px), runtime (ms)
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing import preprocess_image
from edge_detection import detect_edges
from contour_detection import find_document_contour
from perspective import perspective_transform, order_points
from enhancement import global_threshold, adaptive_threshold

ROOT = Path(__file__).resolve().parent.parent
EXT = (".jpg", ".jpeg", ".png", ".bmp")


def load_annotations(path):
    # CSV: filename,condition,x1,y1,x2,y2,x3,y3,x4,y4 (koordinat foto ASLI)
    ann = {}
    if not path.exists():
        return ann
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pts = np.array(
                [[float(row[f"x{i}"]), float(row[f"y{i}"])] for i in range(1, 5)],
                dtype="float32",
            )
            ann[row["filename"]] = {
                "condition": row.get("condition", "").strip() or "unknown",
                "corners": order_points(pts),
            }
    return ann


def draw_result(image, contour):
    vis = image.copy()
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3)
        for (x, y) in order_points(contour).astype(int):
            cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
    return vis


def save_comparison(name, resized, vis_g, vis_m, warped, out_dir):
    fig, axs = plt.subplots(1, 5, figsize=(22, 6))
    items = [
        (cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), "Foto asli"),
        (cv2.cvtColor(vis_g, cv2.COLOR_BGR2RGB), "Corner (Gaussian)"),
        (cv2.cvtColor(vis_m, cv2.COLOR_BGR2RGB), "Corner (Median)"),
    ]
    if warped is not None:
        items += [
            (global_threshold(warped), "Global (Otsu)"),
            (adaptive_threshold(warped), "Adaptive"),
        ]
    for ax, (im, title) in zip(axs, items):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(title)
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_comparison.png", dpi=100)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "dataset" / "raw"))
    ap.add_argument("--annotations",
                    default=str(ROOT / "dataset" / "annotations" / "corners.csv"))
    ap.add_argument("--output", default=str(ROOT / "results"))
    ap.add_argument("--tol", type=float, default=20.0,
                    help="batas corner error (piksel, skala lebar 800) agar dianggap benar")
    ap.add_argument("--canny", choices=["fixed", "auto"], default="fixed",
                    help="fixed = ambang 75/200, auto = dari median intensitas")
    ap.add_argument("--min-area-ratio", type=float, default=None,
                    help="luas kontur minimal sebagai rasio luas gambar, misal 0.2")
    ap.add_argument("--convex", action="store_true",
                    help="hanya terima kontur 4 sudut yang cembung")
    args = ap.parse_args(argv)

    out = Path(args.output)
    img_dir = out / "images"
    cmp_dir = out / "comparison"
    img_dir.mkdir(parents=True, exist_ok=True)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in Path(args.input).iterdir() if p.suffix.lower() in EXT)
    if not files:
        print("Tidak ada foto di", args.input)
        return
    ann = load_annotations(Path(args.annotations))
    if not ann:
        print("Catatan: corners.csv tidak ada, corner error tidak dihitung.")

    rows, thr_rows = [], []

    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {p.name}")
        image = cv2.imread(str(p))
        if image is None:
            print("  dilewati: gambar tidak terbaca")
            continue

        info = ann.get(p.name, {})
        condition = info.get("condition", "unknown")
        gt = info.get("corners")

        resized, gray, gaussian, median = preprocess_image(image)
        scale = resized.shape[1] / image.shape[1]
        gt_scaled = gt * scale if gt is not None else None

        vis = {}
        warped_gauss = None
        warped_any = None

        for fname, filtered in (("gaussian", gaussian), ("median", median)):
            t0 = time.perf_counter()
            edges = detect_edges(filtered, args.canny)
            contour = find_document_contour(edges, args.min_area_ratio, args.convex)
            t_detect = (time.perf_counter() - t0) * 1000

            found = contour is not None
            err = np.nan
            if found and gt_scaled is not None:
                err = float(np.mean(np.linalg.norm(
                    order_points(contour) - gt_scaled, axis=1)))

            runtime_warp = np.nan
            if found:
                t1 = time.perf_counter()
                warped = perspective_transform(resized, contour)
                runtime_warp = (time.perf_counter() - t1) * 1000
                if warped_any is None:
                    warped_any = warped
                if fname == "gaussian":
                    warped_gauss = warped

            rows.append({
                "image": p.name, "condition": condition, "filter": fname,
                "detected": int(found),
                "correct": int(found and not np.isnan(err) and err <= args.tol)
                           if gt_scaled is not None else np.nan,
                "corner_error_px": err,
                "runtime_detect_ms": round(t_detect, 2),
                "runtime_warp_ms": round(runtime_warp, 2) if found else np.nan,
            })
            vis[fname] = draw_result(resized, contour)
            cv2.imwrite(str(img_dir / f"{p.stem}_edges_{fname}.png"), edges)

        if warped_any is not None:
            cv2.imwrite(str(img_dir / f"{p.stem}_scan.png"), warped_any)
            for tname, func in (("global_otsu", global_threshold),
                                ("adaptive", adaptive_threshold)):
                t2 = time.perf_counter()
                res = func(warped_any)
                ms = (time.perf_counter() - t2) * 1000
                cv2.imwrite(str(img_dir / f"{p.stem}_scan_{tname}.png"), res)
                thr_rows.append({"image": p.name, "condition": condition,
                                 "method": tname, "runtime_ms": round(ms, 2)})

        save_comparison(p.stem, resized, vis["gaussian"], vis["median"],
                        warped_any, cmp_dir)

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(out / "metrics_threshold.csv", index=False)

    def agg(g):
        return g.agg(
            jumlah=("image", "count"),
            detection_rate_pct=("detected", lambda s: 100 * s.mean()),
            correct_rate_pct=("correct", lambda s: 100 * s.mean()),
            mean_corner_error_px=("corner_error_px", "mean"),
            mean_runtime_ms=("runtime_detect_ms", "mean"),
        ).round(2).reset_index()

    overall = agg(df.groupby("filter"))
    overall.insert(1, "condition", "SEMUA")
    per_cond = agg(df.groupby(["filter", "condition"]))
    summary = pd.concat([overall, per_cond], ignore_index=True)
    summary.to_csv(out / "metrics_summary.csv", index=False)

    print("\n=== RINGKASAN (Gaussian vs Median) ===")
    print(summary.to_string(index=False))
    if thr_rows:
        print("\n=== RUNTIME THRESHOLDING (ms) ===")
        print(pd.DataFrame(thr_rows).groupby("method")["runtime_ms"]
              .mean().round(2).to_string())

    piv = per_cond.pivot(index="condition", columns="filter",
                         values="detection_rate_pct")
    if not piv.empty:
        ax = piv.plot(kind="bar", figsize=(7, 4))
        ax.set_ylabel("Detection rate (%)")
        ax.set_title("Detection rate: Gaussian vs Median")
        ax.set_ylim(0, 105)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out / "summary_detection_rate.png", dpi=120)
        plt.close()


if __name__ == "__main__":
    main()
