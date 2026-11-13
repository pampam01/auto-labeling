"""
Script Auto-Labeling & Export Roboflow (Universal untuk 3 Benda Apapun)
- Mendeteksi posisi bounding box objek secara otomatis menggunakan `auto_labeler.py`
  (Background-Reference + CIE LAB + Sobel + GrabCut), bukan kotak statis di tengah.
- Menulis file label YOLO (.txt) di samping setiap gambar + `classes.txt` + `data.yaml`.
- Otomatis membuat paket `dataset_roboflow.zip` (split train/valid/test terstratifikasi)
  yang siap langsung di-upload (drag & drop) ke Roboflow!
"""

import os
import re
import glob
import yaml
import shutil
import random
import zipfile
import argparse
import cv2

from auto_labeler import detect_object_bbox, load_background_model

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
ROBOFLOW_EXPORT_DIR = os.path.join(ROOT_DIR, "dataset_roboflow")
ROBOFLOW_ZIP_PATH = os.path.join(ROOT_DIR, "dataset_roboflow.zip")
IGNORED_DIRS = {"preview_test", "roboflow_ready", "__pycache__"}

STAGE_FALLBACK_SCALES = {
    "jepretan1": 0.65,
    "jepretan2": 0.68,
    "jepretan3": 0.68,
    "jepretan4": 0.80,
    "jepretan5": 0.48,
    "jepretan6": 0.65,
}


def _fallback_bbox(filename: str) -> tuple:
    """Fallback aman jika objek sangat menyatu dengan latar dan gagal dideteksi otomatis."""
    scale = 0.65
    for tag, val in STAGE_FALLBACK_SCALES.items():
        if tag in filename.lower():
            scale = val
            break
    return (0.50, 0.50, scale, scale)


def discover_classes(dataset_dir: str, explicit_classes=None) -> list:
    """Menentukan daftar kelas secara dinamis dari argumen, classes.txt, atau subfolder dataset."""
    if explicit_classes:
        return [re.sub(r"[^a-zA-Z0-9_-]+", "_", c.strip().lower()).strip("_") for c in explicit_classes if c.strip()]

    classes_txt = os.path.join(dataset_dir, "classes.txt")
    if os.path.exists(classes_txt):
        with open(classes_txt, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        if classes:
            return classes

    if not os.path.exists(dataset_dir):
        return []

    dirs = [
        d for d in sorted(os.listdir(dataset_dir))
        if os.path.isdir(os.path.join(dataset_dir, d)) and d not in IGNORED_DIRS and not d.startswith(".")
    ]
    return dirs


def write_dataset_metadata(dataset_dir: str, classes: list):
    """Menulis classes.txt dan data.yaml standar YOLO/Roboflow."""
    os.makedirs(dataset_dir, exist_ok=True)
    classes_file = os.path.join(dataset_dir, "classes.txt")
    with open(classes_file, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(f"{c}\n")

    yaml_data = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(classes),
        "names": classes,
    }
    yaml_file = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)


def buat_label_dataset(
    dataset_dir: str = DATASET_DIR,
    classes: list = None,
    overwrite: bool = False,
    use_grabcut: bool = True,
):
    """Menjalankan auto-labeling pada seluruh gambar di folder kelas."""
    classes = discover_classes(dataset_dir, classes)
    if not classes:
        print(f"[ERROR] Tidak ada folder kelas ditemukan di '{dataset_dir}/'.")
        return []

    print("=" * 65)
    print(" AUTO-LABELING BOUNDING BOX YOLO (ADAPTIF UNTUK BENDA APAPUN)")
    print("=" * 65)
    print(f"Kelas terdeteksi ({len(classes)}): {classes}")

    bg_npz = os.path.join(dataset_dir, ".bg_model.npz")
    bg_jpg = os.path.join(dataset_dir, ".bg_ref.jpg")
    bg_model = load_background_model(npz_path=bg_npz, jpg_path=bg_jpg)
    if bg_model is not None:
        print("[INFO] Menggunakan model statistik background (Median+MAD + Horprasert + Saliency).")
    else:
        print("[INFO] Referensi background tidak ditemukan -> memakai mode Border-Contrast + Saliency + GrabCut.")

    write_dataset_metadata(dataset_dir, classes)

    total_dibuat = 0
    total_dilewati = 0
    total_fallback = 0

    for class_id, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_folder):
            print(f"[WARNING] Folder '{class_folder}' tidak ditemukan, dilewati.")
            continue

        files = sorted([
            f for f in os.listdir(class_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        print(f"\nMemproses kelas '{class_name}' (ID: {class_id}) -> {len(files)} foto...")

        class_bg_npz = os.path.join(class_folder, ".bg_model.npz")
        class_bg_jpg = os.path.join(class_folder, ".bg_ref.jpg")
        class_bg_model = load_background_model(npz_path=class_bg_npz, jpg_path=class_bg_jpg) or bg_model

        for filename in files:
            img_path = os.path.join(class_folder, filename)
            txt_path = os.path.join(class_folder, os.path.splitext(filename)[0] + ".txt")

            if os.path.exists(txt_path) and not overwrite:
                # Pastikan file tidak kosong
                with open(txt_path, "r", encoding="utf-8") as f:
                    if f.read().strip():
                        total_dilewati += 1
                        continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            bbox = detect_object_bbox(img, bg_model=class_bg_model, use_grabcut=use_grabcut)
            if bbox is None:
                bbox = _fallback_bbox(filename)
                total_fallback += 1

            nx, ny, nw, nh = bbox
            with open(txt_path, "w", encoding="utf-8") as f_out:
                f_out.write(f"{class_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
            total_dibuat += 1

        print(f" -> Selesai memproses '{class_name}'.")

    print("\n" + "-" * 65)
    print(f"Label baru/diperbarui : {total_dibuat}")
    print(f"Label sudah ada (skip): {total_dilewati}")
    print(f"Fallback box          : {total_fallback}")
    print("-" * 65)
    return classes


def export_roboflow_package(
    dataset_dir: str = DATASET_DIR,
    classes: list = None,
    export_dir: str = ROBOFLOW_EXPORT_DIR,
    zip_path: str = ROBOFLOW_ZIP_PATH,
    train_ratio: float = 0.80,
    valid_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Mengemas dataset menjadi struktur YOLOv8/YOLOv11 (train/valid/test + data.yaml)
    dan mengompresnya menjadi file .zip yang siap drag & drop ke Roboflow.
    """
    classes = discover_classes(dataset_dir, classes)
    if not classes:
        print("[ERROR] Tidak ada kelas untuk di-export ke Roboflow.")
        return None

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    splits = ("train", "valid", "test")
    for sp in splits:
        os.makedirs(os.path.join(export_dir, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(export_dir, sp, "labels"), exist_ok=True)

    rng = random.Random(seed)
    counts = {sp: 0 for sp in splits}

    for class_name in classes:
        class_folder = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_folder):
            continue

        pairs = []
        for fname in sorted(os.listdir(class_folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]
            img_p = os.path.join(class_folder, fname)
            txt_p = os.path.join(class_folder, stem + ".txt")
            if os.path.exists(txt_p):
                pairs.append((img_p, txt_p, fname, stem + ".txt"))

        rng.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        split_map = {
            "train": pairs[:n_train],
            "valid": pairs[n_train:n_train + n_valid],
            "test": pairs[n_train + n_valid:],
        }

        for sp, items in split_map.items():
            for img_p, txt_p, img_name, txt_name in items:
                shutil.copy2(img_p, os.path.join(export_dir, sp, "images", img_name))
                shutil.copy2(txt_p, os.path.join(export_dir, sp, "labels", txt_name))
                counts[sp] += 1

    # Tulis data.yaml untuk Roboflow / YOLO
    yaml_data = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(classes),
        "names": classes,
    }
    with open(os.path.join(export_dir, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)

    with open(os.path.join(export_dir, "classes.txt"), "w", encoding="utf-8") as f:
        for c in classes:
            f.write(f"{c}\n")

    # Kompres ke ZIP
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(export_dir):
            for file in files:
                abs_p = os.path.join(root, file)
                rel_p = os.path.relpath(abs_p, export_dir)
                zf.write(abs_p, rel_p)

    print("\n" + "=" * 65)
    print(f" PAKET ROBOFLOW SIAP UPLOAD: '{zip_path}'")
    print(f"   - Train : {counts['train']} gambar")
    print(f"   - Valid : {counts['valid']} gambar")
    print(f"   - Test  : {counts['test']} gambar")
    print(f"   - Kelas : {classes}")
    print(" Cara upload: Buka Roboflow -> Upload Data -> Drag & drop file")
    print(f" '{zip_path}' (atau folder '{export_dir}/').")
    print("=" * 65)
    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Labeling & Export Roboflow (Universal)")
    parser.add_argument("--dataset", default=DATASET_DIR, help="Folder dataset (default: dataset)")
    parser.add_argument("--classes", nargs="+", default=None, help="Daftar kelas (opsional, otomatis baca dari folder)")
    parser.add_argument("--overwrite", action="store_true", help="Timpa file label .txt yang sudah ada")
    parser.add_argument("--no-grabcut", action="store_true", help="Matikan GrabCut (lebih cepat)")
    parser.add_argument("--no-zip", action="store_true", help="Jangan buat dataset_roboflow.zip")
    args = parser.parse_args()

    detected_classes = buat_label_dataset(
        dataset_dir=args.dataset,
        classes=args.classes,
        overwrite=args.overwrite,
        use_grabcut=not args.no_grabcut,
    )
    if detected_classes and not args.no_zip:
        export_roboflow_package(dataset_dir=args.dataset, classes=detected_classes)
