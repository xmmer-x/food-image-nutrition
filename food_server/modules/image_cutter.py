from pathlib import Path
import csv
import re
import shutil
import time
import uuid

import cv2
import numpy as np
from ultralytics import YOLO, YOLOE


# =========================
# 1. 基础路径
# =========================
SERVER_DIR = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent
MODELS_DIR = SERVER_DIR / "models"

YOLOE_MODEL_NAME = "yoloe-26s-seg.pt"
YOLO26_MODEL_NAME = "yolo26n-seg.pt"


# =========================
# 2. 参数
# =========================
YOLOE_MEAT_PROMPTS = [
    "meat",
    "seafood",
    "beef",
    "pork",
    "raw meat",
    "chicken meat",
]

YOLOE_PRODUCE_PROMPTS = [
    "apple",
    "apricot",
    "avocado",
    "banana",
    "blackberry",
    "blueberry",
    "cantaloupe",
    "cherry",
    "coconut",
    "fig",
    "grape",
    "grapefruit",
    "kiwi",
    "lemon",
    "lime",
    "lychee",
    "mango",
    "orange",
    "papaya",
    "passion fruit",
    "peach",
    "pear",
    "pineapple",
    "plum",
    "pomegranate",
    "raspberry",
    "strawberry",
    "watermelon",
    "tomato",
    "bell pepper",
    "pepper",
    "corn",
    "potato",
    "sweet potato",
    "yam",
    "taro",
    "lotus root",
    "mushroom",
    "shiitake mushroom",
    "cabbage",
    "red cabbage",
    "cauliflower",
    "lettuce",
    "bok choy",
    "chinese cabbage",
    "spinach",
    "cucumber",
    "eggplant",
    "onion",
    "green onion",
    "leek",
    "asparagus",
    "carrot",
    "broccoli",
    "pumpkin",
    "okra",
    "green bean",
    "snap pea",
    "pea",
    "beetroot",
    "artichoke",
    "celery",
]

YOLOE_MEAT_CONF_THRES = 0.15
YOLOE_MEAT_IMGSZ = 640
YOLOE_MEAT_MIN_AREA_RATIO = 0.01

YOLOE_PRODUCE_CONF_THRES = 0.10
YOLOE_PRODUCE_IMGSZ = 960
YOLOE_PRODUCE_MIN_AREA_RATIO = 0.0005

YOLO26_CONF_THRES = 0.25
YOLO26_IMGSZ = 640
YOLO26_MIN_AREA_RATIO = 0.002

IOU_DUP_THRESHOLD = 0.70
CONTAIN_THRESHOLD = 0.85
FORCE_CONTAIN_THRESHOLD = 0.65

SAVE_WHOLE_IMAGE_IF_SINGLE_TYPE = False
WHOLE_IMAGE_MODEL_ID = 0

ADJACENT_GAP_RATIO = 0.05
MAX_PRODUCE_CROP_AREA_RATIO = 0.78
LARGE_PRODUCE_CROP_AREA_RATIO = 0.18
MAX_MEAT_CROP_AREA_RATIO = 0.88
MAX_COMPOSITE_FILL_RATIO = 0.58

MODEL_ID_MAP = {
    "yoloe_meat": 1,
    "yolo26": 2,
    "yoloe_produce": 3,
}

YOLO26_PRODUCE_LABELS = {
    "banana",
    "apple",
    "orange",
    "broccoli",
    "carrot",
}

BACKGROUND_KEYWORDS = [
    "table",
    "dining table",
    "bowl",
    "plate",
    "cup",
    "chair",
    "person",
    "human",
    "hand",
]


# =========================
# 3. 模型懒加载
# =========================
_yoloe_model = None
_yolo26_model = None


def _resolve_model_path(model_name: str) -> str:
    """
    优先使用项目内自带的模型，避免服务器从不同目录启动时找不到文件。
    如果本地都没有，就直接使用模型名，让 ultralytics 自己查找/下载。
    """
    for local_path in (
        MODELS_DIR / model_name,
        SERVER_DIR / model_name,
        MODULE_DIR / model_name,
    ):
        if local_path.exists():
            return str(local_path)

    return model_name


def load_models_once():
    """
    服务器启动后第一次分析图片时加载模型。
    后续请求复用同一个模型，不重复加载。
    """
    global _yoloe_model, _yolo26_model

    if _yoloe_model is None:
        print(f"正在加载 YOLOE：{YOLOE_MODEL_NAME}")
        _yoloe_model = YOLOE(_resolve_model_path(YOLOE_MODEL_NAME))
        print("YOLOE 加载完成")

    if _yolo26_model is None:
        print(f"正在加载 YOLO26n-seg：{YOLO26_MODEL_NAME}")
        _yolo26_model = YOLO(_resolve_model_path(YOLO26_MODEL_NAME))
        print("YOLO26n-seg 加载完成")

    return _yoloe_model, _yolo26_model


# =========================
# 4. 工具函数
# =========================
def natural_key(path_obj: Path):
    parts = re.split(r"(\d+)", path_obj.stem)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    key.append(path_obj.suffix.lower())
    return key


def is_background_label(label: str) -> bool:
    label_lower = label.lower()
    return any(keyword.lower() in label_lower for keyword in BACKGROUND_KEYWORDS)


def should_skip_yolo26_label(label: str) -> bool:
    return is_background_label(label)


def normalize_label(label: str) -> str:
    label = label.strip().lower()

    mapping = {
        "meat": "meat",
        "raw meat": "meat",
        "beef": "meat",
        "pork": "meat",
        "chicken": "meat",
        "chicken meat": "meat",
    }

    return mapping.get(label, label)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0.5).astype(np.uint8) * 255

    if cv2.countNonZero(mask) == 0:
        return mask

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def save_crop_keep_background(original_img: np.ndarray, mask: np.ndarray, save_path: Path):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return False

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    crop = original_img[y1:y2, x1:x2].copy()
    cv2.imwrite(str(save_path), crop)

    return True


def get_bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    return (x1, y1, x2, y2)


def bbox_area(box):
    if box is None:
        return 0

    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_containment_ratio(box_small, box_big):
    if box_small is None or box_big is None:
        return 0.0

    x1, y1, x2, y2 = box_small
    a1, b1, a2, b2 = box_big

    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    small_area = bbox_area(box_small)

    if small_area == 0:
        return 0.0

    return inter_area / small_area


def boxes_touch_or_close(box1, box2, img_shape):
    if box1 is None or box2 is None:
        return False

    h, w = img_shape[:2]
    pad = max(4, int(min(h, w) * ADJACENT_GAP_RATIO))

    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    x11 -= pad
    y11 -= pad
    x12 += pad
    y12 += pad

    x21 -= pad
    y21 -= pad
    x22 += pad
    y22 += pad

    return not (x12 < x21 or x22 < x11 or y12 < y21 or y22 < y11)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray):
    a = mask_a > 0
    b = mask_b > 0

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0, 0, 0

    return inter / union, int(inter), int(union)


def containment_ratio(mask_small: np.ndarray, mask_big: np.ndarray):
    a = mask_small > 0
    b = mask_big > 0

    area_small = a.sum()

    if area_small == 0:
        return 0.0

    inter = np.logical_and(a, b).sum()
    return float(inter / area_small)


def merge_candidates(cands):
    if len(cands) == 1:
        merged = dict(cands[0])
        merged["merged_count"] = 1
        merged["normalized_label"] = normalize_label(merged["label"])
        merged["bbox"] = get_bbox_from_mask(merged["mask"])
        return merged

    base = max(cands, key=lambda c: (c["area"], c["conf"]))

    merged_mask = np.zeros_like(base["mask"], dtype=np.uint8)

    for c in cands:
        merged_mask = np.maximum(merged_mask, c["mask"])

    merged = dict(base)
    merged["mask"] = merged_mask
    merged["area"] = int(cv2.countNonZero(merged_mask))

    image_area = merged_mask.shape[0] * merged_mask.shape[1]
    merged["area_ratio"] = float(merged["area"] / image_area) if image_area > 0 else 0.0
    merged["conf"] = max(c["conf"] for c in cands)
    merged["merged_count"] = len(cands)
    merged["normalized_label"] = normalize_label(base["label"])
    merged["bbox"] = get_bbox_from_mask(merged_mask)

    return merged


def postprocess_same_food_candidates(candidates):
    if not candidates:
        return candidates

    for c in candidates:
        c["normalized_label"] = normalize_label(c["label"])
        c["bbox"] = get_bbox_from_mask(c["mask"])
        c["merged_count"] = c.get("merged_count", 1)

    grouped = {}

    for c in candidates:
        grouped.setdefault(c["normalized_label"], []).append(c)

    final_candidates = []

    for _, group in grouped.items():
        if len(group) == 1:
            final_candidates.append(group[0])
            continue

        n = len(group)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue

            stack = [i]
            visited[i] = True
            cluster_idx = []

            while stack:
                cur = stack.pop()
                cluster_idx.append(cur)

                for j in range(n):
                    if visited[j]:
                        continue

                    if boxes_touch_or_close(group[cur]["bbox"], group[j]["bbox"], group[cur]["mask"].shape):
                        visited[j] = True
                        stack.append(j)

            clusters.append([group[k] for k in cluster_idx])

        merged_clusters = [merge_candidates(cluster) for cluster in clusters]
        merged_clusters = sorted(
            merged_clusters,
            key=lambda c: (c["area"], c["conf"]),
            reverse=True,
        )

        final_candidates.extend(merged_clusters)

    return final_candidates


def is_duplicate(c1: dict, c2: dict):
    iou, _, _ = mask_iou(c1["mask"], c2["mask"])

    mask_contain1 = containment_ratio(c1["mask"], c2["mask"])
    mask_contain2 = containment_ratio(c2["mask"], c1["mask"])

    box1 = c1.get("bbox") or get_bbox_from_mask(c1["mask"])
    box2 = c2.get("bbox") or get_bbox_from_mask(c2["mask"])

    bbox_contain1 = bbox_containment_ratio(box1, box2)
    bbox_contain2 = bbox_containment_ratio(box2, box1)

    contain1 = max(mask_contain1, bbox_contain1)
    contain2 = max(mask_contain2, bbox_contain2)

    dup = (
        iou >= IOU_DUP_THRESHOLD
        or contain1 >= CONTAIN_THRESHOLD
        or contain2 >= CONTAIN_THRESHOLD
    )

    return dup, iou, contain1, contain2


def is_yolo26_produce(c: dict) -> bool:
    return c["model_key"] == "yolo26" and c["label"].lower() in YOLO26_PRODUCE_LABELS


def choose_better(c1: dict, c2: dict, iou: float, contain1: float, contain2: float):
    area1 = c1["area"]
    area2 = c2["area"]

    bg1 = is_background_label(c1["label"])
    bg2 = is_background_label(c2["label"])

    if bg1 != bg2:
        if bg1:
            return c2, c1, f"prefer_non_background(iou={iou:.3f})"
        return c1, c2, f"prefer_non_background(iou={iou:.3f})"

    if contain1 >= CONTAIN_THRESHOLD and area2 > area1 * 1.05:
        return c2, c1, f"containment_always_keep_larger(contain1={contain1:.3f})"

    if contain2 >= CONTAIN_THRESHOLD and area1 > area2 * 1.05:
        return c1, c2, f"containment_always_keep_larger(contain2={contain2:.3f})"

    if {c1["model_key"], c2["model_key"]} == {"yoloe_produce", "yolo26"}:
        if c1["model_key"] == "yolo26":
            return c1, c2, f"yolo26_over_yoloe_produce(iou={iou:.3f})"
        return c2, c1, f"yolo26_over_yoloe_produce(iou={iou:.3f})"

    if c1["model_key"] == "yoloe_meat" and is_yolo26_produce(c2):
        return c2, c1, f"yolo26_produce_over_yoloe_meat(iou={iou:.3f})"

    if c2["model_key"] == "yoloe_meat" and is_yolo26_produce(c1):
        return c1, c2, f"yolo26_produce_over_yoloe_meat(iou={iou:.3f})"

    if {c1["model_key"], c2["model_key"]} == {"yoloe_meat", "yolo26"}:
        if c1["model_key"] == "yoloe_meat":
            return c1, c2, f"yoloe_meat_over_yolo26(iou={iou:.3f})"
        return c2, c1, f"yoloe_meat_over_yolo26(iou={iou:.3f})"

    if abs(c1["conf"] - c2["conf"]) > 1e-6:
        if c1["conf"] > c2["conf"]:
            return c1, c2, f"higher_conf(iou={iou:.3f})"
        return c2, c1, f"higher_conf(iou={iou:.3f})"

    if area1 >= area2:
        return c1, c2, f"larger_area(iou={iou:.3f})"

    return c2, c1, f"larger_area(iou={iou:.3f})"


def dedup_candidates(candidates):
    if not candidates:
        return [], []

    rough_priority = {
        "yolo26": 3,
        "yoloe_meat": 2,
        "yoloe_produce": 1,
    }

    ordered = sorted(
        candidates,
        key=lambda c: (rough_priority[c["model_key"]], c["conf"], c["area"]),
        reverse=True,
    )

    kept = []
    dropped = []

    for cand in ordered:
        should_add = True
        replace_idx = None

        for idx, exist in enumerate(kept):
            dup, iou, contain_cand, contain_exist = is_duplicate(cand, exist)

            if not dup:
                continue

            keep_one, drop_one, reason = choose_better(cand, exist, iou, contain_cand, contain_exist)

            if keep_one is cand and drop_one is exist:
                replace_idx = idx
                dropped.append({
                    **exist,
                    "drop_reason": reason,
                    "kept_against_model": cand["model_key"],
                    "kept_against_label": cand["label"],
                    "dup_iou": round(iou, 4),
                    "dup_contain_self": round(contain_exist, 4),
                    "dup_contain_other": round(contain_cand, 4),
                })
                should_add = True
            else:
                dropped.append({
                    **cand,
                    "drop_reason": reason,
                    "kept_against_model": exist["model_key"],
                    "kept_against_label": exist["label"],
                    "dup_iou": round(iou, 4),
                    "dup_contain_self": round(contain_cand, 4),
                    "dup_contain_other": round(contain_exist, 4),
                })
                should_add = False

            break

        if replace_idx is not None:
            kept[replace_idx] = cand
        elif should_add:
            kept.append(cand)

    return kept, dropped


def force_remove_contained_candidates(candidates):
    if len(candidates) <= 1:
        return candidates, []

    for c in candidates:
        c["bbox"] = c.get("bbox") or get_bbox_from_mask(c["mask"])

    ordered = sorted(
        candidates,
        key=lambda c: (c["area"], c["conf"]),
        reverse=True,
    )

    kept = []
    dropped = []

    for cand in ordered:
        is_dropped = False

        for exist in kept:
            if exist["area"] <= cand["area"] * 1.05:
                continue

            bbox_contain = bbox_containment_ratio(cand["bbox"], exist["bbox"])
            mask_contain = containment_ratio(cand["mask"], exist["mask"])
            contain = max(bbox_contain, mask_contain)

            if contain >= FORCE_CONTAIN_THRESHOLD:
                dropped.append({
                    **cand,
                    "drop_reason": f"force_contained_by_larger(contain={contain:.3f})",
                    "kept_against_model": exist["model_key"],
                    "kept_against_label": exist["label"],
                    "dup_iou": "",
                    "dup_contain_self": round(contain, 4),
                    "dup_contain_other": "",
                })
                is_dropped = True
                break

        if not is_dropped:
            kept.append(cand)

    return kept, dropped


def candidate_crop_area_ratio(candidate):
    bbox = candidate.get("bbox") or get_bbox_from_mask(candidate["mask"])
    if bbox is None:
        return 0.0

    image_area = candidate["mask"].shape[0] * candidate["mask"].shape[1]
    return bbox_area(bbox) / image_area if image_area > 0 else 0.0


def candidate_crop_fill_ratio(candidate):
    bbox = candidate.get("bbox") or get_bbox_from_mask(candidate["mask"])
    if bbox is None:
        return 0.0

    crop_area = bbox_area(bbox)
    return candidate["area"] / crop_area if crop_area > 0 else 0.0


def is_oversized_composite_candidate(candidate):
    crop_area_ratio = candidate_crop_area_ratio(candidate)
    crop_fill_ratio = candidate_crop_fill_ratio(candidate)

    if candidate["model_key"] == "yoloe_produce":
        if crop_area_ratio >= MAX_PRODUCE_CROP_AREA_RATIO:
            return True

        return (
            crop_area_ratio >= LARGE_PRODUCE_CROP_AREA_RATIO
            and crop_fill_ratio <= MAX_COMPOSITE_FILL_RATIO
        )

    if candidate["model_key"] == "yoloe_meat":
        return (
            crop_area_ratio >= MAX_MEAT_CROP_AREA_RATIO
            and crop_fill_ratio <= MAX_COMPOSITE_FILL_RATIO
        )

    return False


def filter_oversized_composite_candidates(candidates):
    return [c for c in candidates if not is_oversized_composite_candidate(c)]


def collect_from_yoloe_group(
    model,
    prompts,
    model_key: str,
    model_id: int,
    annotated_dir: Path,
    conf_thres: float,
    imgsz: int,
    min_area_ratio: float,
    image_path: Path,
    image_idx: int,
):
    model.set_classes(prompts)

    results = model.predict(
        source=str(image_path),
        conf=conf_thres,
        imgsz=imgsz,
        retina_masks=True,
        verbose=False,
    )

    result = results[0]

    annotated_img = result.plot()
    annotated_save_path = annotated_dir / f"{image_path.stem}_{model_key}_annotated.jpg"
    cv2.imwrite(str(annotated_save_path), annotated_img)

    candidates = []

    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        return candidates

    original_img = result.orig_img
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes
    image_area = original_img.shape[0] * original_img.shape[1]

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        label = result.names[cls_id]

        mask = clean_mask(masks[i])
        area = cv2.countNonZero(mask)

        if area == 0:
            continue

        area_ratio = area / image_area

        if area_ratio < min_area_ratio:
            continue

        candidates.append({
            "source_idx": image_idx,
            "source_name": image_path.name,
            "source_stem": image_path.stem,
            "model_key": model_key,
            "model_id": model_id,
            "label": label,
            "conf": conf,
            "mask": mask,
            "area": int(area),
            "area_ratio": float(area_ratio),
            "original_img": original_img,
        })

    return candidates


def collect_from_yolo26(
    model,
    image_path: Path,
    image_idx: int,
    annotated_dir: Path,
):
    results = model.predict(
        source=str(image_path),
        conf=YOLO26_CONF_THRES,
        imgsz=YOLO26_IMGSZ,
        retina_masks=True,
        verbose=False,
    )

    result = results[0]

    annotated_img = result.plot()
    annotated_save_path = annotated_dir / f"{image_path.stem}_yolo26_annotated.jpg"
    cv2.imwrite(str(annotated_save_path), annotated_img)

    candidates = []

    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        return candidates

    original_img = result.orig_img
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes
    image_area = original_img.shape[0] * original_img.shape[1]

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        label = result.names[cls_id]

        if should_skip_yolo26_label(label):
            continue

        mask = clean_mask(masks[i])
        area = cv2.countNonZero(mask)

        if area == 0:
            continue

        area_ratio = area / image_area

        if area_ratio < YOLO26_MIN_AREA_RATIO:
            continue

        candidates.append({
            "source_idx": image_idx,
            "source_name": image_path.name,
            "source_stem": image_path.stem,
            "model_key": "yolo26",
            "model_id": MODEL_ID_MAP["yolo26"],
            "label": label,
            "conf": conf,
            "mask": mask,
            "area": int(area),
            "area_ratio": float(area_ratio),
            "original_img": original_img,
        })

    return candidates


# =========================
# 5. 服务器调用入口
# =========================
def cut_food_image(image_path: Path, output_dir: Path):
    """
    服务器端图片切割入口。

    参数：
    - image_path: 上传后的原图路径
    - output_dir: crops 输出文件夹

    返回：
    {
        "count": 裁剪数量,
        "items": 裁剪结果列表,
        "time": 耗时
    }
    """

    start_time = time.perf_counter()

    output_dir.mkdir(parents=True, exist_ok=True)

    result_root = output_dir.parent
    annotated_yoloe_meat_dir = result_root / "annotated_yoloe_meat"
    annotated_yolo26_dir = result_root / "annotated_yolo26n"
    annotated_yoloe_produce_dir = result_root / "annotated_yoloe_produce"

    annotated_yoloe_meat_dir.mkdir(parents=True, exist_ok=True)
    annotated_yolo26_dir.mkdir(parents=True, exist_ok=True)
    annotated_yoloe_produce_dir.mkdir(parents=True, exist_ok=True)

    report_csv = result_root / "crop_info.csv"
    drop_csv = result_root / "dropped_duplicates.csv"

    yoloe_model, yolo26_model = load_models_once()

    image_idx = 1

    yoloe_meat_candidates = collect_from_yoloe_group(
        model=yoloe_model,
        prompts=YOLOE_MEAT_PROMPTS,
        model_key="yoloe_meat",
        model_id=MODEL_ID_MAP["yoloe_meat"],
        annotated_dir=annotated_yoloe_meat_dir,
        conf_thres=YOLOE_MEAT_CONF_THRES,
        imgsz=YOLOE_MEAT_IMGSZ,
        min_area_ratio=YOLOE_MEAT_MIN_AREA_RATIO,
        image_path=image_path,
        image_idx=image_idx,
    )

    yolo26_candidates = collect_from_yolo26(
        model=yolo26_model,
        image_path=image_path,
        image_idx=image_idx,
        annotated_dir=annotated_yolo26_dir,
    )

    yoloe_produce_candidates = collect_from_yoloe_group(
        model=yoloe_model,
        prompts=YOLOE_PRODUCE_PROMPTS,
        model_key="yoloe_produce",
        model_id=MODEL_ID_MAP["yoloe_produce"],
        annotated_dir=annotated_yoloe_produce_dir,
        conf_thres=YOLOE_PRODUCE_CONF_THRES,
        imgsz=YOLOE_PRODUCE_IMGSZ,
        min_area_ratio=YOLOE_PRODUCE_MIN_AREA_RATIO,
        image_path=image_path,
        image_idx=image_idx,
    )

    all_candidates = filter_oversized_composite_candidates(
        yoloe_meat_candidates + yolo26_candidates + yoloe_produce_candidates
    )

    kept, dropped = dedup_candidates(all_candidates)
    kept = filter_oversized_composite_candidates(kept)

    kept = postprocess_same_food_candidates(kept)
    kept = filter_oversized_composite_candidates(kept)

    kept, force_dropped = force_remove_contained_candidates(kept)
    kept = filter_oversized_composite_candidates(kept)

    dropped_rows = dropped + force_dropped

    kept_sorted = sorted(
        kept,
        key=lambda c: (c["model_id"], -c["area"], -c["conf"]),
    )

    crop_items = []
    final_rows = []

    normalized_label_set = {normalize_label(c["label"]) for c in kept_sorted}

    # 没有检测到任何食材时，服务器版做一个兜底：
    # 保存整张原图，避免后续流程没有输入。
    if len(kept_sorted) == 0:
        fallback_name = f"{image_path.stem}_whole_no_detection{image_path.suffix.lower()}"
        fallback_path = output_dir / fallback_name
        shutil.copy2(image_path, fallback_path)

        crop_items.append({
            "crop_name": fallback_name,
            "crop_path": str(fallback_path),
            "label": "no_detection",
            "normalized_label": "no_detection",
            "confidence": 0.0,
            "model_id": WHOLE_IMAGE_MODEL_ID,
            "model_name": "whole_image",
            "area": "",
            "area_ratio": "",
            "merged_count": 1,
        })

        final_rows.append([
            fallback_name,
            image_idx,
            image_path.name,
            WHOLE_IMAGE_MODEL_ID,
            "whole_image",
            "no_detection",
            "no_detection",
            "",
            "",
            "",
            1,
        ])

    elif (
        SAVE_WHOLE_IMAGE_IF_SINGLE_TYPE
        and len(normalized_label_set) == 1
    ):
        only_type = next(iter(normalized_label_set))
        whole_name = f"{image_path.stem}_whole_single_type{image_path.suffix.lower()}"
        whole_path = output_dir / whole_name
        shutil.copy2(image_path, whole_path)

        crop_items.append({
            "crop_name": whole_name,
            "crop_path": str(whole_path),
            "label": f"single_type:{only_type}",
            "normalized_label": only_type,
            "confidence": 0.0,
            "model_id": WHOLE_IMAGE_MODEL_ID,
            "model_name": "whole_image",
            "area": "",
            "area_ratio": "",
            "merged_count": 1,
        })

        final_rows.append([
            whole_name,
            image_idx,
            image_path.name,
            WHOLE_IMAGE_MODEL_ID,
            "whole_image",
            f"single_type:{only_type}",
            only_type,
            "",
            "",
            "",
            1,
        ])

    else:
        per_model_counter = {1: 0, 2: 0, 3: 0}

        for cand in kept_sorted:
            model_id = cand["model_id"]
            per_model_counter[model_id] += 1
            seq = per_model_counter[model_id]

            out_name = f"{image_path.stem}-{model_id}-{seq}.png"
            out_path = output_dir / out_name

            saved = save_crop_keep_background(
                cand["original_img"],
                cand["mask"],
                out_path,
            )

            if not saved:
                continue

            item = {
                "crop_name": out_name,
                "crop_path": str(out_path),
                "label": cand["label"],
                "normalized_label": cand.get("normalized_label", normalize_label(cand["label"])),
                "confidence": round(cand["conf"], 4),
                "model_id": cand["model_id"],
                "model_name": cand["model_key"],
                "area": cand["area"],
                "area_ratio": round(cand["area_ratio"], 6),
                "merged_count": cand.get("merged_count", 1),
            }

            crop_items.append(item)

            final_rows.append([
                out_name,
                image_idx,
                image_path.name,
                cand["model_id"],
                cand["model_key"],
                cand["label"],
                cand.get("normalized_label", normalize_label(cand["label"])),
                round(cand["conf"], 4),
                cand["area"],
                round(cand["area_ratio"], 6),
                cand.get("merged_count", 1),
            ])

    with report_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name",
            "source_image_index",
            "source_image_name",
            "model_id",
            "model_name",
            "label",
            "normalized_label",
            "confidence",
            "mask_area",
            "area_ratio",
            "merged_count",
        ])
        writer.writerows(final_rows)

    with drop_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_image_index",
            "source_image_name",
            "dropped_model",
            "dropped_label",
            "dropped_confidence",
            "dropped_area",
            "drop_reason",
            "kept_against_model",
            "kept_against_label",
            "dup_iou",
            "dup_contain_self",
            "dup_contain_other",
        ])

        for row in dropped_rows:
            writer.writerow([
                row["source_idx"],
                row["source_name"],
                row["model_key"],
                row["label"],
                round(row["conf"], 4),
                row["area"],
                row["drop_reason"],
                row["kept_against_model"],
                row["kept_against_label"],
                row["dup_iou"],
                row["dup_contain_self"],
                row["dup_contain_other"],
            ])

    time_used = round(time.perf_counter() - start_time, 3)

    return {
        "count": len(crop_items),
        "items": crop_items,
        "time": time_used,
    }
