from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

COLOR_FAT = (255, 255, 255)       # 白色：脂肪，BGR
COLOR_LEAN = (40, 40, 230)        # 红色：瘦肉，BGR
COLOR_UNKNOWN = (160, 160, 160)   # 灰色：不确定，BGR
COLOR_CONTOUR = (0, 255, 255)     # 黄色边线，BGR


def read_image_any(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"图片读取失败：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"图片保存失败：{path}")
    encoded.tofile(str(path))


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"不支持的图片格式：{input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"路径不存在：{input_path}")

    images = sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"没有找到图片：{input_path}")
    return images


def split_bgr_alpha(image: np.ndarray, alpha_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """优先用透明PNG的alpha通道获得肉类区域。"""
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        meat_mask = image > 0
        return bgr, meat_mask

    if image.shape[2] == 4:
        bgr = image[:, :, :3].copy()
        alpha = image[:, :, 3]
        meat_mask = alpha > alpha_threshold
        return bgr, meat_mask

    bgr = image[:, :, :3].copy()

    # 没有alpha时的兜底方案：排除接近纯黑背景。
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    meat_mask = gray > 5
    return bgr, meat_mask


def erode_mask(mask: np.ndarray, erode_pixels: int) -> np.ndarray:
    if erode_pixels <= 0:
        return mask

    kernel_size = erode_pixels * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(bool)


def make_features(bgr: np.ndarray, meat_mask: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """生成Lab/HSV/RGB差值混合特征。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    pixels_bgr = bgr[meat_mask].astype(np.float32)
    pixels_hsv = hsv[meat_mask].astype(np.float32)
    pixels_lab = lab[meat_mask].astype(np.float32)

    b = pixels_bgr[:, 0]
    g = pixels_bgr[:, 1]
    r = pixels_bgr[:, 2]

    l = pixels_lab[:, 0]
    a = pixels_lab[:, 1] - 128.0
    bb = pixels_lab[:, 2] - 128.0

    s = pixels_hsv[:, 1]
    v = pixels_hsv[:, 2]

    red_minus_green = (r - g) / 255.0
    red_minus_blue = (r - b) / 255.0

    features = np.stack(
        [
            (l / 255.0) * 1.00,
            (a / 128.0) * 1.45,
            (bb / 128.0) * 0.75,
            (s / 255.0) * 1.15,
            (v / 255.0) * 0.45,
            red_minus_green * 0.85,
            red_minus_blue * 0.65,
        ],
        axis=1,
    ).astype(np.float32)

    raw = {"bgr": pixels_bgr, "hsv": pixels_hsv, "lab": pixels_lab}
    return features, raw


def run_kmeans(features: np.ndarray, cluster_count: int, max_sample_pixels: int, seed: int) -> np.ndarray:
    total = features.shape[0]
    if total == 0:
        return np.empty((0,), dtype=np.int32)

    k = min(cluster_count, total)
    if k <= 1:
        return np.zeros((total,), dtype=np.int32)

    rng = np.random.default_rng(seed)

    if total > max_sample_pixels:
        sample_indices = rng.choice(total, size=max_sample_pixels, replace=False)
        sample = features[sample_indices]
    else:
        sample = features

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.03)

    _, _, centers = cv2.kmeans(
        sample.astype(np.float32),
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = np.empty((total,), dtype=np.int32)
    centers = centers.astype(np.float32)
    chunk_size = 200_000

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = features[start:end]
        distances = np.sum((chunk[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels[start:end] = np.argmin(distances, axis=1).astype(np.int32)

    return labels


def classify_cluster(mean_bgr: np.ndarray, mean_hsv: np.ndarray, mean_lab: np.ndarray, args: argparse.Namespace):
    b, g, r = [float(x) for x in mean_bgr]
    h, s, v = [float(x) for x in mean_hsv]
    l = float(mean_lab[0])
    a_pos = float(mean_lab[1] - 128.0)
    b_pos = float(mean_lab[2] - 128.0)

    red_green = max(0.0, r - g)
    red_blue = max(0.0, r - b)

    # 脂肪倾向：亮、低饱和、不太红、白/米白/浅黄。
    fat_score = (
        0.45 * (l / 255.0)
        + 0.35 * (1.0 - s / 255.0)
        + 0.22 * (1.0 - min(max(a_pos, 0.0) / 55.0, 1.0))
        + 0.08 * min(max(b_pos, 0.0) / 45.0, 1.0)
    )

    # 瘦肉倾向：偏红、饱和度较高、R明显大于G/B。
    lean_score = (
        0.46 * min(max(a_pos, 0.0) / 55.0, 1.0)
        + 0.28 * (s / 255.0)
        + 0.16 * min(red_green / 120.0, 1.0)
        + 0.10 * min(red_blue / 120.0, 1.0)
    )

    if v < args.min_valid_v or l < args.min_valid_l:
        label = "unknown"
    elif (
        fat_score >= lean_score + args.fat_margin
        and l >= args.min_fat_l
        and s <= args.max_fat_s
        and a_pos <= args.max_fat_a
    ):
        label = "fat"
    elif (
        lean_score >= fat_score - args.lean_tolerance
        or (a_pos >= args.force_lean_a and s >= args.force_lean_s)
    ):
        label = "lean"
    else:
        label = "unknown"

    info = {
        "mean_b": b,
        "mean_g": g,
        "mean_r": r,
        "mean_h": h,
        "mean_s": s,
        "mean_v": v,
        "mean_l": l,
        "mean_a_centered": a_pos,
        "mean_b_centered": b_pos,
        "fat_score": fat_score,
        "lean_score": lean_score,
    }
    return label, info


def analyze_one_image(image_path: Path, args: argparse.Namespace, output_root: Path):
    image = read_image_any(image_path)
    bgr, meat_mask = split_bgr_alpha(image, alpha_threshold=args.alpha_threshold)
    meat_mask = erode_mask(meat_mask, erode_pixels=args.erode)

    h, w = bgr.shape[:2]
    meat_total = int(np.count_nonzero(meat_mask))

    fat_mask = np.zeros((h, w), dtype=np.uint8)
    lean_mask = np.zeros((h, w), dtype=np.uint8)
    unknown_mask = np.zeros((h, w), dtype=np.uint8)
    cluster_rows = []

    if meat_total == 0:
        return {
            "file_name": image_path.name,
            "meat_pixels": 0,
            "fat_pixels": 0,
            "lean_pixels": 0,
            "unknown_pixels": 0,
            "fat_ratio_excluding_unknown": 0.0,
            "lean_ratio_excluding_unknown": 0.0,
            "unknown_ratio_inside_meat": 1.0,
            "status": "empty_alpha_mask",
        }, cluster_rows

    features, raw = make_features(bgr, meat_mask)
    labels = run_kmeans(
        features=features,
        cluster_count=args.clusters,
        max_sample_pixels=args.max_sample_pixels,
        seed=args.seed,
    )

    meat_indices = np.flatnonzero(meat_mask.reshape(-1))
    fat_flat = fat_mask.reshape(-1)
    lean_flat = lean_mask.reshape(-1)
    unknown_flat = unknown_mask.reshape(-1)

    for cluster_id in sorted(set(int(x) for x in labels.tolist())):
        cluster_pixels = labels == cluster_id
        pixel_count = int(np.count_nonzero(cluster_pixels))
        if pixel_count == 0:
            continue

        mean_bgr = raw["bgr"][cluster_pixels].mean(axis=0)
        mean_hsv = raw["hsv"][cluster_pixels].mean(axis=0)
        mean_lab = raw["lab"][cluster_pixels].mean(axis=0)

        cluster_label, info = classify_cluster(mean_bgr, mean_hsv, mean_lab, args)
        global_indices = meat_indices[cluster_pixels]

        if cluster_label == "fat":
            fat_flat[global_indices] = 255
        elif cluster_label == "lean":
            lean_flat[global_indices] = 255
        else:
            unknown_flat[global_indices] = 255

        cluster_rows.append(
            {
                "file_name": image_path.name,
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "pixel_count": pixel_count,
                "pixel_ratio_inside_meat": pixel_count / meat_total,
                **info,
            }
        )

    fat_pixels = int(np.count_nonzero(fat_mask))
    lean_pixels = int(np.count_nonzero(lean_mask))
    unknown_pixels = int(np.count_nonzero(unknown_mask))

    valid_pixels = fat_pixels + lean_pixels
    fat_ratio = fat_pixels / valid_pixels if valid_pixels > 0 else 0.0
    lean_ratio = lean_pixels / valid_pixels if valid_pixels > 0 else 0.0
    unknown_ratio = unknown_pixels / max(meat_total, 1)

    stem = image_path.stem

    write_image(output_root / "masks_fat" / f"{stem}_fat.png", fat_mask)
    write_image(output_root / "masks_lean" / f"{stem}_lean.png", lean_mask)
    write_image(output_root / "masks_unknown" / f"{stem}_unknown.png", unknown_mask)

    visualization = np.zeros_like(bgr)
    visualization[unknown_mask > 0] = COLOR_UNKNOWN
    visualization[lean_mask > 0] = COLOR_LEAN
    visualization[fat_mask > 0] = COLOR_FAT

    meat_u8 = meat_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(meat_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(visualization, contours, -1, COLOR_CONTOUR, 2)

    write_image(output_root / "visualizations" / f"{stem}_fat_lean_visual.png", visualization)

    overlay = bgr.copy()
    area = meat_mask
    overlay[area] = cv2.addWeighted(bgr[area], 0.45, visualization[area], 0.55, 0)
    cv2.drawContours(overlay, contours, -1, COLOR_CONTOUR, 2)

    text = (
        f"fat {fat_ratio * 100:.1f}% | "
        f"lean {lean_ratio * 100:.1f}% | "
        f"unknown {unknown_ratio * 100:.1f}%"
    )
    cv2.putText(overlay, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(overlay, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 1, cv2.LINE_AA)

    write_image(output_root / "overlays" / f"{stem}_overlay.jpg", overlay)

    return {
        "file_name": image_path.name,
        "meat_pixels": meat_total,
        "fat_pixels": fat_pixels,
        "lean_pixels": lean_pixels,
        "unknown_pixels": unknown_pixels,
        "fat_ratio_excluding_unknown": fat_ratio,
        "lean_ratio_excluding_unknown": lean_ratio,
        "unknown_ratio_inside_meat": unknown_ratio,
        "status": "ok",
    }, cluster_rows


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="不训练模型，直接根据透明肉类切割图的像素颜色，估算脂肪/瘦肉面积比例。"
    )

    parser.add_argument("input_path", type=Path, help="输入：单张透明PNG，或cutouts文件夹")
    parser.add_argument("--output", type=Path, default=None, help="输出文件夹。默认在输入旁边生成 fat_lean_results")
    parser.add_argument("--clusters", type=int, default=5, help="颜色聚类数量。默认5，常用范围4～7")
    parser.add_argument("--alpha-threshold", type=int, default=10, help="alpha大于该值才算肉类区域。默认10")
    parser.add_argument("--erode", type=int, default=0, help="分析前腐蚀肉类mask边缘的像素数。默认0")
    parser.add_argument("--max-sample-pixels", type=int, default=200000, help="KMeans最多采样多少肉类像素。默认200000")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。默认42")

    # 分类规则参数。一般不用改；结果偏差明显时再调。
    parser.add_argument("--fat-margin", type=float, default=0.08, help="越大越保守地判定脂肪")
    parser.add_argument("--lean-tolerance", type=float, default=0.05, help="越大越容易判定瘦肉")
    parser.add_argument("--min-fat-l", type=float, default=128.0, help="脂肪最低Lab亮度")
    parser.add_argument("--max-fat-s", type=float, default=150.0, help="脂肪最高HSV饱和度")
    parser.add_argument("--max-fat-a", type=float, default=45.0, help="脂肪最高Lab红色程度")
    parser.add_argument("--force-lean-a", type=float, default=24.0, help="明显瘦肉的Lab红色程度")
    parser.add_argument("--force-lean-s", type=float, default=45.0, help="明显瘦肉的HSV饱和度")
    parser.add_argument("--min-valid-v", type=float, default=25.0, help="低于此HSV亮度判定为unknown")
    parser.add_argument("--min-valid-l", type=float, default=30.0, help="低于此Lab亮度判定为unknown")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()

    if args.output is None:
        if input_path.is_file():
            output_root = input_path.parent / f"{input_path.stem}_fat_lean_results"
        else:
            output_root = input_path.parent / "fat_lean_results"
    else:
        output_root = args.output.resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    image_paths = collect_images(input_path)

    print("=" * 72)
    print("脂肪 / 瘦肉比例分析：颜色聚类版")
    print(f"输入：      {input_path}")
    print(f"输出：      {output_root}")
    print(f"图片数量：  {len(image_paths)}")
    print(f"聚类数量：  {args.clusters}")
    print("=" * 72)

    summary_rows = []
    all_cluster_rows = []
    start_time = time.time()

    for index, image_path in enumerate(image_paths, start=1):
        try:
            summary, cluster_rows = analyze_one_image(image_path, args, output_root)
            summary_rows.append(summary)
            all_cluster_rows.extend(cluster_rows)

            print(
                f"[{index}/{len(image_paths)}] {image_path.name}  "
                f"fat={summary['fat_ratio_excluding_unknown'] * 100:.1f}%  "
                f"lean={summary['lean_ratio_excluding_unknown'] * 100:.1f}%  "
                f"unknown={summary['unknown_ratio_inside_meat'] * 100:.1f}%"
            )

        except Exception as exc:
            print(f"[失败] {image_path.name}: {exc}")
            summary_rows.append(
                {
                    "file_name": image_path.name,
                    "meat_pixels": 0,
                    "fat_pixels": 0,
                    "lean_pixels": 0,
                    "unknown_pixels": 0,
                    "fat_ratio_excluding_unknown": 0.0,
                    "lean_ratio_excluding_unknown": 0.0,
                    "unknown_ratio_inside_meat": 0.0,
                    "status": f"failed: {exc}",
                }
            )

    save_csv(output_root / "fat_lean_summary.csv", summary_rows)
    save_csv(output_root / "cluster_report.csv", all_cluster_rows)

    elapsed = time.time() - start_time

    print("\n" + "=" * 72)
    print("分析完成")
    print(f"耗时：          {elapsed:.1f}s")
    print(f"汇总CSV：       {output_root / 'fat_lean_summary.csv'}")
    print(f"聚类报告：      {output_root / 'cluster_report.csv'}")
    print(f"可视化图：      {output_root / 'visualizations'}")
    print(f"叠加预览图：    {output_root / 'overlays'}")
    print(f"脂肪Mask：      {output_root / 'masks_fat'}")
    print(f"瘦肉Mask：      {output_root / 'masks_lean'}")
    print(f"不确定Mask：    {output_root / 'masks_unknown'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
