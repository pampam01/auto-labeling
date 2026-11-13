"""
Script Pengambilan Dataset & Auto-Labeling Otomatis dari Webcam (Universal 3 Benda Apapun)
Fitur utama:
  - Bebas menentukan 3 benda apapun via CLI (--classes a b c) atau input interaktif.
  - Kalibrasi Background (tombol B) untuk deteksi objek super akurat & tahan bayangan.
  - Preview Bounding Box real-time langsung di jendela kamera.
  - Filter kualitas otomatis: skip frame blur & frame kosong (tanpa objek) & frame duplikat statis.
  - Langsung menyimpan gambar 512x512 (.jpg) + label YOLO (.txt) secara bersamaan.
  - Otomatis mengemas hasil ke `dataset_roboflow.zip` (train/valid/test + data.yaml) siap upload ke Roboflow.
"""

import os
import re
import time
import argparse
import cv2
import numpy as np

from auto_labeler import (
    detect_object_bbox,
    compute_sharpness,
    compute_frame_difference,
    build_background_model,
    save_background_model,
    load_background_model,
    smooth_bbox,
)
from generate_labels import write_dataset_metadata, export_roboflow_package

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOTAL_IMAGES_TARGET = 1002  # 334 foto per kelas (habis dibagi 3, >= 1000 foto)
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
CAPTURE_INTERVAL = 0.25    # Jeda antar capture otomatis (detik)
MIN_SHARPNESS = 35.0       # Ambang ketajaman Laplacian minimum
MIN_FRAME_DIFF = 1.2       # Perubahan minimum antar frame agar pose bervariasi

VARIATION_STAGES = [
    {
        "id": 1,
        "name": "normal",
        "label": "Jepretan 1/6: Posisi Depan / Normal",
        "instruction": "Posisikan objek di tengah kotak, jarak standar",
    },
    {
        "id": 2,
        "name": "sudut_horizontal",
        "label": "Jepretan 2/6: Sudut Samping Kiri & Kanan (Yaw)",
        "instruction": "Putar/miringkan objek perlahan ke kiri & kanan",
    },
    {
        "id": 3,
        "name": "sudut_vertikal",
        "label": "Jepretan 3/6: Sudut Atas & Bawah (Tilt)",
        "instruction": "Miringkan objek ke atas dan ke bawah perlahan",
    },
    {
        "id": 4,
        "name": "jarak_dekat",
        "label": "Jepretan 4/6: Jarak Dekat / Zoom (Close-Up)",
        "instruction": "Dekatkan objek ke kamera agar tampak detail",
    },
    {
        "id": 5,
        "name": "jarak_jauh",
        "label": "Jepretan 5/6: Jarak Jauh (Far / Full View)",
        "instruction": "Jauhkan objek agar terlihat utuh beserta latar",
    },
    {
        "id": 6,
        "name": "pencahayaan_rotasi",
        "label": "Jepretan 6/6: Pencahayaan & Rotasi Bebas",
        "instruction": "Putar berbagai sudut & ubah arah pencahayaan",
    },
]


def sanitize_class_names(raw_names: list) -> list:
    """Membersihkan nama kelas agar aman sebagai nama folder dan label Roboflow."""
    cleaned = []
    for name in raw_names:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
        if slug and slug not in cleaned:
            cleaned.append(slug)
    return cleaned


def minta_input_kelas() -> list:
    """Meminta user memasukkan 3 nama benda jika tidak diberikan via argumen CLI."""
    print("\n" + "=" * 70)
    print(" SETUP 3 BENDA / KELAS DATASET")
    print("=" * 70)
    while True:
        raw = input("Masukkan nama 3 benda (pisahkan dengan spasi, contoh: botol kacamata mouse): ").strip()
        names = sanitize_class_names(raw.replace(",", " ").split())
        if len(names) == 3:
            return names
        print(f"[INFO] Terdeteksi {len(names)} nama ({names}). Mohon masukkan tepat 3 nama benda yang berbeda.")


def hitung_target_merata(total: int, jumlah_bagian: int) -> list:
    base = total // jumlah_bagian
    sisa = total % jumlah_bagian
    return [base + (1 if i < sisa else 0) for i in range(jumlah_bagian)]


def get_crop_coords(frame_shape, crop_size=512):
    h, w = frame_shape[:2]
    cx = w // 2
    cy = (h // 2) + 15
    y1 = max(0, cy - crop_size // 2)
    y2 = y1 + crop_size
    if y2 > h:
        y2 = h
        y1 = max(0, y2 - crop_size)
    x1 = max(0, cx - crop_size // 2)
    x2 = x1 + crop_size
    if x2 > w:
        x2 = w
        x1 = max(0, x2 - crop_size)
    return x1, y1, x2, y2


def potong_tengah_512(frame, target_size=512):
    x1, y1, x2, y2 = get_crop_coords(frame.shape, target_size)
    crop = frame[y1:y2, x1:x2]
    if crop.shape[:2] != (target_size, target_size):
        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return crop


def simpan_frame_dan_label(crop_512, class_dir, class_id, class_name, count, stage_idx, stage_name, bbox):
    """Menyimpan gambar 512x512 (.jpg) dan label YOLO (.txt) sekaligus."""
    stem = f"{class_name}_{count + 1:04d}_jepretan{stage_idx + 1}_{stage_name}"
    img_path = os.path.join(class_dir, f"{stem}.jpg")
    txt_path = os.path.join(class_dir, f"{stem}.txt")

    cv2.imwrite(img_path, crop_512)
    nx, ny, nw, nh = bbox
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
    return img_path


def gambar_hud(
    frame,
    class_name,
    class_idx,
    total_classes,
    count,
    target,
    stage_count,
    stage_target,
    stage_info,
    is_capturing,
    bbox,
    sharpness,
    has_bg_ref,
    status_hint="",
    countdown=0,
):
    h, w, _ = frame.shape
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, 122), (18, 18, 18), -1)
    cv2.rectangle(overlay, (0, h - 90), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Header kelas & total progress
    title_text = f"KELAS [{class_idx + 1}/{total_classes}]: {class_name.upper()}"
    cv2.putText(frame, title_text, (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.78, (0, 255, 255), 2, cv2.LINE_AA)

    pct = min(100, int((count / max(1, target)) * 100))
    total_progress_text = f"Total: {count}/{target} ({pct}%)"
    cv2.putText(frame, total_progress_text, (w - 280, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = 20, 44, w - 40, 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
    filled_w = min(bar_w, int((count / max(1, target)) * bar_w))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 210, 0), -1)

    sesi_text = f"{stage_info['label']}  [{stage_count}/{stage_target} foto]"
    cv2.putText(frame, sesi_text, (20, 78), cv2.FONT_HERSHEY_DUPLEX, 0.62, (80, 255, 120), 2, cv2.LINE_AA)

    if is_capturing:
        rec_text = "[AUTO-CAPTURE AKTIF - SPASI: Pause]"
        rec_color = (0, 70, 255)
    else:
        rec_text = "[PAUSED - SPASI: Mulai Auto-Capture]"
        rec_color = (0, 220, 220)
    cv2.putText(frame, rec_text, (20, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rec_color, 2, cv2.LINE_AA)

    bg_badge = "BG: TERKALIBRASI" if has_bg_ref else "BG: AUTO (Tekan B saat kosong)"
    bg_color = (0, 255, 120) if has_bg_ref else (0, 200, 255)
    cv2.putText(frame, bg_badge, (w - 310, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.52, bg_color, 2, cv2.LINE_AA)

    # Panduan bawah
    cv2.putText(frame, f"Panduan: {stage_info['instruction']}", (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    shortcuts = "[SPASI: Auto] [C: Manual] [B: Set BG] [R: Reset BG] [S: Skip Sesi] [N: Next] [Q: Keluar]"
    cv2.putText(frame, shortcuts, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)

    # Area crop 512x512
    c_x1, c_y1, c_x2, c_y2 = get_crop_coords(frame.shape, 512)
    cv2.rectangle(frame, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Crop 512x512 | Tajam: {int(sharpness)}", (c_x1 + 8, c_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

    # Gambar Live Auto-Label Bounding Box di dalam area crop
    cw = c_x2 - c_x1
    ch = c_y2 - c_y1
    if bbox is not None:
        nx, ny, nw, nh = bbox
        bx1 = int(c_x1 + (nx - nw / 2.0) * cw)
        by1 = int(c_y1 + (ny - nh / 2.0) * ch)
        bx2 = int(c_x1 + (nx + nw / 2.0) * cw)
        by2 = int(c_y1 + (ny + nh / 2.0) * ch)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"AUTO-BOX: {class_name}", (bx1, max(c_y1 + 40, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Objek belum terdeteksi di area crop", (c_x1 + 70, c_y1 + ch // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2, cv2.LINE_AA)

    if status_hint:
        cv2.putText(frame, status_hint, (c_x1 + 10, c_y2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    if countdown > 0:
        cd_str = str(countdown)
        ts = cv2.getTextSize(cd_str, cv2.FONT_HERSHEY_DUPLEX, 3.5, 5)[0]
        cv2.putText(frame, cd_str, ((w - ts[0]) // 2, (h + ts[1]) // 2), cv2.FONT_HERSHEY_DUPLEX, 3.5, (0, 255, 255), 5, cv2.LINE_AA)

    return frame


def tampilkan_layar_jeda_sesi(cap, next_stage_info, finished_stage_label, finished_count):
    while True:
        ret, frame = cap.read()
        if not ret:
            return False
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        overlay = frame.copy()
        card_w, card_h = min(860, w - 60), 230
        card_x, card_y = (w - card_w) // 2, (h - card_h) // 2
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (25, 25, 25), -1)
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (0, 255, 255), 2)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        cv2.putText(frame, f"[SELESAI] {finished_stage_label} ({finished_count} foto)", (card_x + 30, card_y + 45), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"BERIKUTNYA -> {next_stage_info['label']}", (card_x + 30, card_y + 95), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Instruksi: {next_stage_info['instruction']}", (card_x + 30, card_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, "Tekan SPASI / ENTER untuk lanjut sesi ini (Q untuk keluar)", (card_x + 30, card_y + 195), cv2.FONT_HERSHEY_DUPLEX, 0.6, (50, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow("Pengambilan Dataset & Auto-Labeling", frame)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord(" "), 13):
            return True
        if k in (27, ord("q")):
            return False


def jalankan_pengambilan_gambar(
    camera_id=0,
    total_target=TOTAL_IMAGES_TARGET,
    class_list=None,
    dataset_dir=DATASET_DIR,
    interval=CAPTURE_INTERVAL,
    min_sharpness=MIN_SHARPNESS,
    min_diff=MIN_FRAME_DIFF,
    auto_export=True,
):
    if not class_list:
        class_list = minta_input_kelas()
    else:
        class_list = sanitize_class_names(class_list)

    if len(class_list) == 0:
        print("[ERROR] Daftar kelas kosong.")
        return

    os.makedirs(dataset_dir, exist_ok=True)
    write_dataset_metadata(dataset_dir, class_list)
    targets = hitung_target_merata(total_target, len(class_list))

    bg_npz_path = os.path.join(dataset_dir, ".bg_model.npz")
    bg_ref_path = os.path.join(dataset_dir, ".bg_ref.jpg")
    bg_model = load_background_model(npz_path=bg_npz_path, jpg_path=bg_ref_path)

    print("\n" + "=" * 70)
    print(" PENGAMBILAN DATASET & AUTO-LABELING REAL-TIME (UNIVERSAL)")
    print("=" * 70)
    print(f"Kelas ({len(class_list)}): {class_list}")
    print(f"Target Total : {total_target} gambar")
    print("Tips Akurasi : Kosongkan area kotak kuning lalu tekan [B] untuk")
    print("               kalibrasi background agar auto-labeling presisi!")
    print("=" * 70)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka kamera ID {camera_id}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    win_name = "Pengambilan Dataset & Auto-Labeling"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    for class_idx, class_name in enumerate(class_list):
        class_target = targets[class_idx]
        class_dir = os.path.join(dataset_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        existing_imgs = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        total_class_count = len(existing_imgs)
        remaining_target = max(0, class_target - total_class_count)

        if remaining_target == 0:
            print(f"[SKIP] Kelas '{class_name}' sudah mencapai target ({total_class_count}/{class_target}).")
            continue

        stage_targets = hitung_target_merata(remaining_target, len(VARIATION_STAGES))
        last_saved_crop = None

        for stage_idx, stage_info in enumerate(VARIATION_STAGES):
            stage_target = stage_targets[stage_idx]
            if stage_target == 0:
                continue

            if stage_idx > 0:
                prev_stage = VARIATION_STAGES[stage_idx - 1]
                if not tampilkan_layar_jeda_sesi(cap, stage_info, prev_stage["label"], stage_targets[stage_idx - 1]):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            stage_count = 0
            is_capturing = False
            last_capture_time = 0.0
            status_hint = ""
            prev_bbox = None

            while stage_count < stage_target:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                crop_512 = potong_tengah_512(frame, 512)

                # Deteksi bounding box real-time + smoothing temporal + ketajaman
                raw_bbox = detect_object_bbox(crop_512, bg_model=bg_model, use_grabcut=False)
                bbox = smooth_bbox(prev_bbox, raw_bbox, alpha=0.65)
                prev_bbox = bbox
                sharpness = compute_sharpness(crop_512)

                now = time.time()
                status_hint = ""
                if is_capturing and (now - last_capture_time >= interval):
                    if bbox is None:
                        status_hint = "SKIP: Objek tidak terdeteksi"
                    elif sharpness < min_sharpness:
                        status_hint = "SKIP: Gambar blur, tahan stabil"
                    elif last_saved_crop is not None and compute_frame_difference(crop_512, last_saved_crop) < min_diff:
                        status_hint = "INFO: Gerakkan/putar objek sedikit"
                    else:
                        simpan_frame_dan_label(
                            crop_512, class_dir, class_idx, class_name,
                            total_class_count, stage_idx, stage_info["name"], bbox
                        )
                        last_saved_crop = crop_512.copy()
                        stage_count += 1
                        total_class_count += 1
                        last_capture_time = now
                        if stage_count >= stage_target:
                            is_capturing = False
                            break

                disp = gambar_hud(
                    frame=frame,
                    class_name=class_name,
                    class_idx=class_idx,
                    total_classes=len(class_list),
                    count=total_class_count,
                    target=class_target,
                    stage_count=stage_count,
                    stage_target=stage_target,
                    stage_info=stage_info,
                    is_capturing=is_capturing,
                    bbox=bbox,
                    sharpness=sharpness,
                    has_bg_ref=(bg_model is not None),
                    status_hint=status_hint,
                )
                cv2.imshow(win_name, disp)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord("b"):
                    # Ambil 10 frame untuk model statistik Median+MAD agar tahan noise & flicker
                    bg_frames = [crop_512.copy()]
                    for _ in range(9):
                        rb, fb = cap.read()
                        if rb:
                            fb = cv2.flip(fb, 1)
                            bg_frames.append(potong_tengah_512(fb, 512))
                        cv2.waitKey(25)
                    bg_model = build_background_model(bg_frames)
                    save_background_model(bg_model, bg_npz_path, bg_ref_path)
                    prev_bbox = None
                    print("[OK] Model statistik background (10 frame Median+MAD) berhasil dikalibrasi!")
                elif key == ord("r"):
                    bg_model = None
                    prev_bbox = None
                    for p in (bg_npz_path, bg_ref_path):
                        if os.path.exists(p):
                            os.remove(p)
                    print("[OK] Referensi background di-reset ke mode AUTO!")
                elif key == ord(" "):
                    is_capturing = not is_capturing
                    if is_capturing:
                        for sec in range(3, 0, -1):
                            r2, f2 = cap.read()
                            if r2:
                                f2 = cv2.flip(f2, 1)
                                c2 = potong_tengah_512(f2, 512)
                                b2 = detect_object_bbox(c2, bg_model=bg_model, use_grabcut=False)
                                cd = gambar_hud(
                                    f2, class_name, class_idx, len(class_list),
                                    total_class_count, class_target, stage_count, stage_target,
                                    stage_info, True, b2, compute_sharpness(c2),
                                    bg_model is not None, countdown=sec
                                )
                                cv2.imshow(win_name, cd)
                                cv2.waitKey(700)
                        last_capture_time = time.time()
                elif key == ord("c"):
                    save_box = bbox if bbox is not None else (0.50, 0.50, 0.65, 0.65)
                    simpan_frame_dan_label(
                        crop_512, class_dir, class_idx, class_name,
                        total_class_count, stage_idx, stage_info["name"], save_box
                    )
                    last_saved_crop = crop_512.copy()
                    stage_count += 1
                    total_class_count += 1
                elif key == ord("s"):
                    break
                elif key == ord("n"):
                    total_class_count = class_target
                    break

            if total_class_count >= class_target:
                break

        print(f"[SUKSES] Kelas '{class_name}' selesai ({total_class_count} foto + label).")
        if class_idx < len(class_list) - 1:
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    ov = frame.copy()
                    cv2.rectangle(ov, (0, h // 2 - 65), (w, h // 2 + 65), (15, 15, 15), -1)
                    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
                    cv2.putText(frame, f"KELAS '{class_name.upper()}' SELESAI!", (40, h // 2 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 0), 2)
                    cv2.putText(frame, "Siapkan benda berikutnya, lalu tekan SPASI / ENTER", (40, h // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                    cv2.imshow(win_name, frame)
                k = cv2.waitKey(30) & 0xFF
                if k in (ord(" "), 13):
                    break
                if k in (27, ord("q")):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()

    if auto_export:
        export_roboflow_package(dataset_dir=dataset_dir, classes=class_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture Dataset & Auto-Labeling Webcam (Universal 3 Benda)")
    parser.add_argument("--camera", type=int, default=0, help="ID Webcam (default: 0)")
    parser.add_argument("--target", type=int, default=TOTAL_IMAGES_TARGET, help="Total target gambar keseluruhan (default: 1002 = 334/kelas)")
    parser.add_argument("--classes", nargs="+", default=None, help="Nama 3 benda (contoh: --classes botol kacamata mouse)")
    parser.add_argument("--dataset", default=DATASET_DIR, help="Folder output dataset (default: dataset)")
    parser.add_argument("--interval", type=float, default=CAPTURE_INTERVAL, help="Jeda capture otomatis dalam detik (default: 0.25)")
    parser.add_argument("--min-sharpness", type=float, default=MIN_SHARPNESS, help="Ambang ketajaman minimum (default: 35.0)")
    parser.add_argument("--min-diff", type=float, default=MIN_FRAME_DIFF, help="Perbedaan minimum antar frame (default: 1.2)")
    parser.add_argument("--no-export", action="store_true", help="Jangan langsung kemas ke dataset_roboflow.zip setelah selesai")
    args = parser.parse_args()

    jalankan_pengambilan_gambar(
        camera_id=args.camera,
        total_target=args.target,
        class_list=args.classes,
        dataset_dir=args.dataset,
        interval=args.interval,
        min_sharpness=args.min_sharpness,
        min_diff=args.min_diff,
        auto_export=not args.no_export,
    )
