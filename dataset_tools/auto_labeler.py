"""
Core Auto-Labeling Engine (Research-Backed & Class-Agnostic untuk 3 Benda Apapun)
Menggabungkan 4 teknik Computer Vision klasik & modern agar akurat di kondisi nyata:
1. Multi-Frame Statistical Background (Median + MAD):
   Menghilangkan noise kamera per piksel dengan mengambil median & deviasi dari beberapa frame.
2. Illumination & White-Balance Alignment:
   Menormalkan pergeseran kecerahan/eksposur otomatis webcam sebelum membandingkan frame.
3. Horprasert Shadow Suppression + Structural Gradient Diff:
   Membedakan bayangan (brightness turun tapi kromatisitas & tekstur tetap) dari objek nyata.
4. Hou & Zhang (CVPR 2007) Spectral Residual Saliency + Hysteresis Thresholding:
   Menonjolkan objek yang kontrasnya mirip background sekaligus membuang bintik noise di latar.
"""

import cv2
import numpy as np

PROC_SIZE = 256


def compute_sharpness(image_bgr: np.ndarray) -> float:
    """Menghitung tingkat ketajaman gambar menggunakan Laplacian variance."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_frame_difference(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Menghitung perbedaan rata-rata antar dua frame untuk mencegah duplikat statis."""
    small_a = cv2.resize(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), (32, 32))
    small_b = cv2.resize(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY), (32, 32))
    return float(np.mean(np.abs(small_a.astype(np.float32) - small_b.astype(np.float32))))


def build_background_model(frames_bgr: list) -> dict:
    """
    Membangun model statistik background (per-pixel Median & MAD) dari beberapa frame kosong.
    Jauh lebih tahan noise webcam dibanding 1 foto statis.
    """
    labs = []
    grays = []
    for f in frames_bgr:
        small = cv2.resize(f, (PROC_SIZE, PROC_SIZE), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (5, 5), 0)
        labs.append(cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32))
        grays.append(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY).astype(np.float32))

    lab_stack = np.stack(labs, axis=0)
    bg_median_lab = np.median(lab_stack, axis=0).astype(np.float32)
    bg_mad_lab = np.median(np.abs(lab_stack - bg_median_lab[None, ...]), axis=0).astype(np.float32)
    # Lantai noise minimum agar area statis tidak terlalu sensitif
    bg_mad_lab = np.maximum(bg_mad_lab, 2.5)

    gray_median = np.median(np.stack(grays, axis=0), axis=0).astype(np.float32)
    gx = cv2.Sobel(gray_median, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_median, cv2.CV_32F, 0, 1, ksize=3)
    bg_grad = np.hypot(gx, gy)

    # Preview BGR dari median LAB untuk disimpan sebagai .bg_ref.jpg
    bg_bgr_med = cv2.cvtColor(np.clip(bg_median_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return {
        "median_lab": bg_median_lab,
        "mad_lab": bg_mad_lab,
        "grad": bg_grad,
        "bgr": bg_bgr_med,
    }


def save_background_model(model: dict, npz_path: str, jpg_path: str = None):
    """Menyimpan model statistik background ke disk (.npz) dan opsional preview (.jpg)."""
    np.savez_compressed(
        npz_path,
        median_lab=model["median_lab"],
        mad_lab=model["mad_lab"],
        grad=model["grad"],
    )
    if jpg_path:
        cv2.imwrite(jpg_path, model["bgr"])


def load_background_model(npz_path: str = None, jpg_path: str = None) -> dict | None:
    """Memuat model statistik background (.npz) atau fallback dari 1 gambar (.jpg)."""
    import os
    if npz_path and os.path.exists(npz_path):
        data = np.load(npz_path)
        return {
            "median_lab": data["median_lab"].astype(np.float32),
            "mad_lab": data["mad_lab"].astype(np.float32),
            "grad": data["grad"].astype(np.float32),
        }
    if jpg_path and os.path.exists(jpg_path):
        img = cv2.imread(jpg_path)
        if img is not None:
            return build_background_model([img])
    return None


def _spectral_residual_saliency(gray_f32: np.ndarray) -> np.ndarray:
    """
    Hou & Zhang (CVPR 2007): Spectral Residual Saliency.
    Mendeteksi objek yang menonjol secara visual lewat residual log-spektrum FFT (~2 ms).
    """
    small = cv2.resize(gray_f32, (64, 64), interpolation=cv2.INTER_AREA)
    fft = np.fft.fft2(small)
    amp = np.abs(fft) + 1e-8
    log_amp = np.log(amp)
    phase = np.angle(fft)

    avg_log_amp = cv2.blur(log_amp, (3, 3))
    spectral_residual = log_amp - avg_log_amp

    sal_fft = np.exp(spectral_residual + 1j * phase)
    sal_map = np.abs(np.fft.ifft2(sal_fft)) ** 2
    sal_map = cv2.GaussianBlur(sal_map.astype(np.float32), (9, 9), 2.5)
    sal_map = cv2.resize(sal_map, (PROC_SIZE, PROC_SIZE), interpolation=cv2.INTER_LINEAR)

    s_max = float(sal_map.max())
    if s_max > 1e-6:
        sal_map = (sal_map / s_max) * 255.0
    return sal_map.astype(np.float32)


def _align_illumination(frame_lab: np.ndarray, bg_lab: np.ndarray) -> np.ndarray:
    """
    Menyelaraskan pencahayaan & white-balance global antara frame saat ini dan background
    menggunakan tepi luar frame (border 14px) yang umumnya merupakan area latar.
    Mengatasi auto-exposure / flicker webcam!
    """
    b = 14
    mask_border = np.zeros((PROC_SIZE, PROC_SIZE), dtype=bool)
    mask_border[:b, :] = True
    mask_border[-b:, :] = True
    mask_border[:, :b] = True
    mask_border[:, -b:] = True

    aligned = frame_lab.copy()
    for ch in range(3):
        f_vals = frame_lab[:, :, ch][mask_border]
        bg_vals = bg_lab[:, :, ch][mask_border]
        f_med = float(np.median(f_vals))
        bg_med = float(np.median(bg_vals))
        # Hanya koreksi bias moderat agar objek besar yang menyentuh tepi tidak merusak
        shift = np.clip(bg_med - f_med, -25.0, 25.0)
        aligned[:, :, ch] = np.clip(aligned[:, :, ch] + shift, 0.0, 255.0)
    return aligned


def _hysteresis_threshold(score_map: np.ndarray, high_val: float, low_val: float) -> np.ndarray:
    """
    Hysteresis thresholding (dual-threshold region growing):
    Hanya area lemah (low) yang terhubung langsung ke inti objek kuat (high) yang dipertahankan.
    Bintik noise terpisah di background otomatis terbuang!
    """
    strong = (score_map >= high_val).astype(np.uint8)
    weak = (score_map >= low_val).astype(np.uint8)

    if np.count_nonzero(strong) == 0:
        return np.zeros_like(strong, dtype=np.uint8)

    num_labels, labels = cv2.connectedComponents(weak, connectivity=8)
    strong_labels = np.unique(labels[strong > 0])
    strong_labels = strong_labels[strong_labels > 0]
    if len(strong_labels) == 0:
        return np.zeros_like(strong, dtype=np.uint8)

    mask = np.isin(labels, strong_labels).astype(np.uint8) * 255
    return mask


def _refine_with_grabcut(small_bgr: np.ndarray, fg_mask: np.ndarray, iters: int = 2) -> np.ndarray:
    """Memperhalus mask foreground menggunakan GrabCut pada citra 256x256."""
    h, w = small_bgr.shape[:2]
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    pad_y, pad_x = max(2, int(h * 0.04)), max(2, int(w * 0.04))
    gc_mask[:pad_y, :] = cv2.GC_BGD
    gc_mask[-pad_y:, :] = cv2.GC_BGD
    gc_mask[:, :pad_x] = cv2.GC_BGD
    gc_mask[:, -pad_x:] = cv2.GC_BGD

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_fg = cv2.erode(fg_mask, kernel, iterations=1)
    gc_mask[fg_mask > 0] = cv2.GC_PR_FGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    if np.count_nonzero(gc_mask == cv2.GC_FGD) < 20 or np.count_nonzero(gc_mask == cv2.GC_BGD) < 20:
        return fg_mask

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small_bgr, gc_mask, None, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_MASK)
        return np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        return fg_mask


def _extract_best_bbox(mask: np.ndarray, min_area_ratio: float = 0.015, max_area_ratio: float = 0.94):
    """Mengekstrak bounding box terbaik (cx, cy, w, h) ternormalisasi dari mask."""
    h, w = mask.shape[:2]
    total_area = float(h * w)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cx_frame, cy_frame = w / 2.0, h / 2.0
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < total_area * min_area_ratio:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cnt_cx = x + bw / 2.0
        cnt_cy = y + bh / 2.0
        dist = np.hypot((cnt_cx - cx_frame) / w, (cnt_cy - cy_frame) / h)
        score = area * max(0.25, 1.0 - dist * 1.15)
        candidates.append((score, area, x, y, x + bw, y + bh))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, main_area, x1, y1, x2, y2 = candidates[0]

    # Gabungkan komponen sekunder yang dekat dengan objek utama
    for _, area, ox1, oy1, ox2, oy2 in candidates[1:]:
        if area < main_area * 0.15:
            continue
        gap_x = max(0, max(x1, ox1) - min(x2, ox2))
        gap_y = max(0, max(y1, oy1) - min(y2, oy2))
        if gap_x < w * 0.08 and gap_y < h * 0.08:
            x1 = min(x1, ox1)
            y1 = min(y1, oy1)
            x2 = max(x2, ox2)
            y2 = max(y2, oy2)

    box_area_ratio = ((x2 - x1) * (y2 - y1)) / total_area
    if box_area_ratio < min_area_ratio or box_area_ratio > max_area_ratio:
        return None

    pad_x = int((x2 - x1) * 0.015)
    pad_y = int((y2 - y1) * 0.015)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    nx = max(0.01, min(0.99, ((x1 + x2) / 2.0) / w))
    ny = max(0.01, min(0.99, ((y1 + y2) / 2.0) / h))
    nw = max(0.02, min(0.98, (x2 - x1) / w))
    nh = max(0.02, min(0.98, (y2 - y1) / h))
    return (nx, ny, nw, nh)


def smooth_bbox(prev_bbox: tuple | None, new_bbox: tuple | None, alpha: float = 0.65) -> tuple | None:
    """Temporal Exponential Moving Average (EMA) untuk menghaluskan kotak saat live webcam."""
    if new_bbox is None:
        return None
    if prev_bbox is None:
        return new_bbox
    return tuple(alpha * n + (1.0 - alpha) * p for p, n in zip(prev_bbox, new_bbox))


def detect_object_bbox(
    image_bgr: np.ndarray,
    bg_bgr: np.ndarray = None,
    bg_model: dict = None,
    use_grabcut: bool = False,
) -> tuple | None:
    """
    Mendeteksi bounding box objek utama pada citra BGR (512x512).
    Dapat menerima `bg_model` (hasil `build_background_model`) atau `bg_bgr` (1 gambar),
    atau berjalan dalam mode standalone jika keduanya None.
    """
    small = cv2.resize(image_bgr, (PROC_SIZE, PROC_SIZE), interpolation=cv2.INTER_AREA)
    small_blur = cv2.GaussianBlur(small, (5, 5), 0)
    lab = cv2.cvtColor(small_blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(small_blur, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)

    # Prior lembut ke area tengah (mengurangi gangguan tepi/sudut latar)
    yy, xx = np.mgrid[0:PROC_SIZE, 0:PROC_SIZE]
    dist_center = np.hypot((xx - PROC_SIZE / 2.0) / PROC_SIZE, (yy - PROC_SIZE / 2.0) / PROC_SIZE)
    center_prior = np.clip(1.12 - dist_center * 0.75, 0.50, 1.12)

    # Saliency FFT untuk membantu objek yang warnanya mirip background
    saliency = _spectral_residual_saliency(gray)

    if bg_model is None and bg_bgr is not None:
        bg_model = build_background_model([bg_bgr])

    if bg_model is not None:
        bg_lab = bg_model["median_lab"]
        bg_mad = bg_model["mad_lab"]
        bg_grad = bg_model["grad"]

        # 1. Normalisasi pencahayaan/white-balance global
        lab_aligned = _align_illumination(lab, bg_lab)

        # 2. Z-score terhadap model statistik background (Median + MAD)
        dL = (lab_aligned[:, :, 0] - bg_lab[:, :, 0]) / (bg_mad[:, :, 0] * 1.5)
        da = np.abs(lab_aligned[:, :, 1] - bg_lab[:, :, 1]) / (bg_mad[:, :, 1] * 1.1)
        db = np.abs(lab_aligned[:, :, 2] - bg_lab[:, :, 2]) / (bg_mad[:, :, 2] * 1.1)

        chroma_score = np.hypot(da, db)
        edge_diff = np.abs(grad - bg_grad) / 14.0

        # 3. Horprasert Shadow Suppression:
        # Jika kecerahan turun (dL < 0) TAPI kromatisitas & struktur tepi tidak berubah -> bayangan!
        is_shadow = (dL < -1.2) & (chroma_score < 2.2) & (edge_diff < 1.6)
        light_score = np.where(is_shadow, np.abs(dL) * 0.15, np.abs(dL) * 0.65)

        # 4. Gabungkan cue: Chromaticity + Lightness + Edge Structure + Spectral Saliency
        raw_score = (light_score + 1.35 * chroma_score + 0.85 * edge_diff + 0.015 * saliency) * center_prior
        score_norm = np.clip(raw_score * 8.0, 0, 255).astype(np.uint8)

        otsu_t, _ = cv2.threshold(score_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_t = max(46.0, float(otsu_t) * 1.05)
        low_t = max(24.0, high_t * 0.52)
        mask = _hysteresis_threshold(score_norm, high_t, low_t)
    else:
        # Mode Standalone: Border-Contrast LAB + Gradient + Spectral Saliency
        border = 18
        top = lab[:border, :, :].reshape(-1, 3)
        bottom = lab[-border:, :, :].reshape(-1, 3)
        left = lab[:, :border, :].reshape(-1, 3)
        right = lab[:, -border:, :].reshape(-1, 3)
        border_pixels = np.vstack([top, bottom, left, right])

        bg_med = np.median(border_pixels, axis=0)
        bg_mad = np.maximum(np.median(np.abs(border_pixels - bg_med), axis=0), 3.0)

        diff = np.abs(lab - bg_med) / bg_mad
        color_score = 0.55 * diff[:, :, 0] + 1.20 * np.hypot(diff[:, :, 1], diff[:, :, 2])
        grad_smooth = cv2.GaussianBlur(grad, (7, 7), 0) / 16.0

        raw_score = (color_score + 0.75 * grad_smooth + 0.025 * saliency) * center_prior
        score_norm = np.clip(raw_score * 10.0, 0, 255).astype(np.uint8)

        otsu_t, _ = cv2.threshold(score_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_t = max(42.0, float(otsu_t) * 1.05)
        low_t = max(22.0, high_t * 0.52)
        mask = _hysteresis_threshold(score_norm, high_t, low_t)

    # Pembersihan morfologi
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    if use_grabcut and np.count_nonzero(mask) > 50:
        mask = _refine_with_grabcut(small, mask, iters=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_open)

    return _extract_best_bbox(mask)
