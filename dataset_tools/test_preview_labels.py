"""
Script Verifikasi & Preview Visual Label Dataset (Universal untuk Benda Apapun)
- Otomatis mendeteksi seluruh kelas di folder `dataset/`.
- Memeriksa kelengkapan & validitas koordinat seluruh pasangan gambar (.jpg) + label (.txt).
- Membuat gambar preview bounding box per kelas + 1 gambar `grid_summary.jpg` di `dataset/preview_test/`.
"""

import os
import random
import argparse
import cv2
import numpy as np

from generate_labels import discover_classes

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
PREVIEW_DIR = os.path.join(DATASET_DIR, "preview_test")
COLORS = [
    (255, 80, 0),    # Biru
    (0, 210, 0),     # Hijau
    (0, 140, 255),   # Oranye
    (200, 0, 200),   # Ungu
]


def draw_yolo_boxes(img: np.ndarray, txt_path: str, classes: list) -> np.ndarray:
    h, w = img.shape[:2]
    vis = img.copy()
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        nx, ny, nw, nh = [float(v) for v in parts[1:5]]
        x1 = max(0, int((nx - nw / 2.0) * w))
        y1 = max(0, int((ny - nh / 2.0) * h))
        x2 = min(w - 1, int((nx + nw / 2.0) * w))
        y2 = min(h - 1, int((ny + nh / 2.0) * h))

        color = COLORS[cid % len(COLORS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

        label_name = classes[cid] if 0 <= cid < len(classes) else f"cls_{cid}"
        label_text = f"{label_name} (ID:{cid})"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(vis, label_text, (x1 + 5, max(th + 2, y1 - 5)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
    return vis


def verifikasi_dan_preview(dataset_dir: str = DATASET_DIR, samples_per_class: int = 2):
    classes = discover_classes(dataset_dir)
    if not classes:
        print(f"[ERROR] Tidak ada kelas ditemukan di '{dataset_dir}/'.")
        return

    preview_dir = os.path.join(dataset_dir, "preview_test")
    os.makedirs(preview_dir, exist_ok=True)

    print("=" * 65)
    print(" VERIFIKASI LABEL DATASET & PREVIEW BOUNDING BOX")
    print("=" * 65)
    print(f"Kelas terdeteksi ({len(classes)}): {classes}\n")

    grid_tiles = []
    total_imgs = 0
    total_missing_txt = 0
    total_invalid = 0

    for cid, cname in enumerate(classes):
        class_dir = os.path.join(dataset_dir, cname)
        if not os.path.exists(class_dir):
            continue

        imgs = sorted([f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        valid_pairs = []

        for fname in imgs:
            total_imgs += 1
            txt_p = os.path.join(class_dir, os.path.splitext(fname)[0] + ".txt")
            if not os.path.exists(txt_p):
                total_missing_txt += 1
                continue
            with open(txt_p, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                total_invalid += 1
                continue
            parts = content.splitlines()[0].split()
            if len(parts) < 5:
                total_invalid += 1
                continue
            valid_pairs.append((os.path.join(class_dir, fname), txt_p, fname))

        print(f"Kelas '{cname}' (ID:{cid}): {len(imgs)} gambar | {len(valid_pairs)} label valid")

        if valid_pairs:
            # Ambil sampel merata (awal, tengah/akhir)
            step = max(1, len(valid_pairs) // samples_per_class)
            chosen = [valid_pairs[i] for i in range(0, len(valid_pairs), step)][:samples_per_class]
            for idx, (img_p, txt_p, fname) in enumerate(chosen):
                img = cv2.imread(img_p)
                if img is None:
                    continue
                vis = draw_yolo_boxes(img, txt_p, classes)
                out_p = os.path.join(preview_dir, f"preview_{cname}_{idx + 1}.jpg")
                cv2.imwrite(out_p, vis)
                grid_tiles.append(cv2.resize(vis, (320, 320)))

    if grid_tiles:
        cols = min(3, len(grid_tiles))
        rows = (len(grid_tiles) + cols - 1) // cols
        blank = np.zeros((320, 320, 3), dtype=np.uint8)
        while len(grid_tiles) < rows * cols:
            grid_tiles.append(blank)
        row_imgs = [np.hstack(grid_tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
        grid = np.vstack(row_imgs)
        grid_path = os.path.join(preview_dir, "grid_summary.jpg")
        cv2.imwrite(grid_path, grid)
        print(f"\n[OK] Grid preview tersimpan di: '{grid_path}'")

    print("-" * 65)
    print(f"Total Gambar        : {total_imgs}")
    print(f"Label Hilang        : {total_missing_txt}")
    print(f"Label Tidak Valid   : {total_invalid}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifikasi & Preview Visual Label YOLO")
    parser.add_argument("--dataset", default=DATASET_DIR, help="Folder dataset (default: dataset)")
    parser.add_argument("--samples", type=int, default=2, help="Jumlah sampel preview per kelas (default: 2)")
    args = parser.parse_args()
    verifikasi_dan_preview(dataset_dir=args.dataset, samples_per_class=args.samples)
