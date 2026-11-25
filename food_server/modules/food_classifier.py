from pathlib import Path
import json
import re
import time

import numpy as np
from PIL import Image


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent.parent
MODEL_DIR = PROJECT_DIR / "模型"

try:
    from modules.nutrition_analyzer import lookup_japanese_name
except Exception:  # pragma: no cover - depends on how the server is launched
    try:
        from .nutrition_analyzer import lookup_japanese_name
    except Exception:
        def lookup_japanese_name(candidates):
            return ""


def _first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path

    return paths[0]

MODEL_CONFIGS = {
    "fruit": {
        "model_name": "水果模型",
        "status": "ready",
        "path": _first_existing_path(
            MODEL_DIR / "fruit.pth",
            MODEL_DIR / "best_efficientnet_b0_fruit_combined_balanced(3).pth",
            MODULE_DIR / "fruit.pth",
        ),
        "class_names_path": _first_existing_path(
            MODEL_DIR / "fruit.json",
            MODEL_DIR / "names_fruit.json",
            MODEL_DIR / "class_names_fruit.json",
            MODULE_DIR / "class_names_fruit.json",
        ),
        "architecture": "efficientnet_b0",
    },
    "vegetable": {
        "model_name": "蔬菜模型",
        "status": "ready",
        "path": MODULE_DIR / "vegetable.pth",
        "architecture": "resnet50",
    },
    "mushroom": {
        "model_name": "蘑菇模型",
        "status": "ready",
        "path": _first_existing_path(
            MODEL_DIR / "mushroom.pth",
            MODEL_DIR / "best_efficientnet_b0_topup.pth",
            MODULE_DIR / "mushroom.pth",
        ),
        "class_names_path": _first_existing_path(
            MODEL_DIR / "mushroom.json",
            MODEL_DIR / "class_names_mushrooms.json",
            MODULE_DIR / "class_names_mushrooms.json",
        ),
        "architecture": "efficientnet_b0",
    },
    "meat": {
        "model_name": "肉类模型",
        "status": "ready",
        "path": MODULE_DIR / "meat.pth",
        "architecture": "meat_resnet50_texture",
    },
    "seafood": {
        "model_name": "海鲜模型",
        "status": "ready",
        "path": MODULE_DIR / "seafood_resnet50_best.pth",
        "architecture": "seafood_resnet50_dropout",
    },
}


try:
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on server runtime
    class _MissingNN:
        Module = object

    cv2 = None
    torch = None
    nn = _MissingNN()
    F = None
    models = None
    transforms = None
    _IMPORT_ERROR = str(exc)


_loaded_models = {}


def classify_cut_items(cut_items):
    """
    对每个切割图执行食材识别。

    返回的新 item 会增加：
    - recognition_results: 每个模型的结果，包含未接入占位
    - best_recognition: 可用于 App 图片下方展示的最高置信度结果
    """
    start = time.perf_counter()
    new_items = []

    for item in cut_items:
        new_item = dict(item)
        crop_path = Path(str(new_item.get("crop_path", "")))
        results = classify_image(crop_path)
        best = choose_best_result(results)

        new_item["recognition_results"] = results
        new_item["best_recognition"] = best

        if best is not None:
            new_item["food_name"] = best.get("display_label") or best.get("label", "")
            new_item["food_confidence"] = best.get("confidence", 0.0)
            new_item["food_model_key"] = best.get("model_key", "")
            new_item["food_model_name"] = best.get("model_name", "")
        else:
            new_item["food_name"] = "未识别"
            new_item["food_confidence"] = 0.0
            new_item["food_model_key"] = ""
            new_item["food_model_name"] = ""

        new_items.append(new_item)

    return {
        "items": new_items,
        "time": round(time.perf_counter() - start, 3),
    }


def classify_image(image_path: Path):
    results = []

    for model_key in ["fruit", "vegetable", "mushroom", "meat", "seafood"]:
        config = MODEL_CONFIGS[model_key]

        if config["status"] != "ready":
            results.append(_not_ready_result(model_key, config))
            continue

        results.append(_predict_with_model(model_key, image_path))

    return results


GENERAL_MODEL_KEYS = {"fruit", "vegetable"}
SPECIALIST_MIN_CONFIDENCE = {
    "mushroom": 0.75,
    "meat": 0.70,
    "seafood": 0.97,
}
GENERAL_MIN_CONFIDENCE = 0.45


def choose_best_result(results):
    ok_results = [r for r in results if r.get("status") == "ok"]

    if not ok_results:
        return None

    positive_results = [r for r in ok_results if _is_positive_label(r.get("label", ""))]

    if not positive_results:
        return max(ok_results, key=lambda r: float(r.get("confidence") or 0.0))

    general_results = [
        r for r in positive_results
        if r.get("model_key") in GENERAL_MODEL_KEYS
        and float(r.get("confidence") or 0.0) >= GENERAL_MIN_CONFIDENCE
    ]

    if general_results:
        return max(general_results, key=lambda r: float(r.get("confidence") or 0.0))

    specialist_results = [
        r for r in positive_results
        if float(r.get("confidence") or 0.0)
        >= SPECIALIST_MIN_CONFIDENCE.get(str(r.get("model_key") or ""), 0.0)
    ]

    if specialist_results:
        return max(specialist_results, key=lambda r: float(r.get("confidence") or 0.0))

    return max(positive_results, key=lambda r: float(r.get("confidence") or 0.0))


def _is_positive_label(label: str) -> bool:
    label_lower = label.strip().lower()

    if not label_lower:
        return False

    negative_labels = {
        "other",
        "unknown",
        "unknown (非肉类或图片不清晰)",
        "no_detection",
        "未识别",
    }

    return label_lower not in negative_labels


MEAT_SEGMENTATION_LABELS = {
    "meat",
    "raw meat",
    "beef",
    "pork",
    "chicken",
    "chicken meat",
}

SEAFOOD_SEGMENTATION_LABELS = {
    "seafood",
    "fish",
    "clam",
    "cod",
    "crab",
    "eel",
    "lobster",
    "mackerel",
    "octopus",
    "oyster",
    "salmon",
    "scallop",
    "shrimp",
    "squid",
    "tuna",
}

MUSHROOM_SEGMENTATION_LABELS = {
    "mushroom",
    "shiitake",
    "shiitake mushroom",
    "agaricus bisporus",
    "boletus",
    "enoki",
    "king oyster mushroom",
    "oyster mushroom",
    "pleurotus eryngii",
}

FRUIT_SEGMENTATION_LABELS = {
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
    "fruit",
    "grape",
    "grapefruit",
    "kiwi",
    "kiwi fruit",
    "lemon",
    "lime",
    "lychee",
    "mango",
    "melon",
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
}

VEGETABLE_SEGMENTATION_LABELS = {
    "artichoke",
    "asparagus",
    "beetroot",
    "bell pepper",
    "bok choy",
    "broccoli",
    "cabbage",
    "carrot",
    "cauliflower",
    "celery",
    "chinese cabbage",
    "corn",
    "cucumber",
    "eggplant",
    "green bean",
    "green onion",
    "lettuce",
    "leek",
    "lotus root",
    "okra",
    "onion",
    "pea",
    "pepper",
    "pimento",
    "potato",
    "pumpkin",
    "red cabbage",
    "snap pea",
    "spinach",
    "sweet potato",
    "taro",
    "tomato",
    "vegetable",
    "yam",
}


def _preferred_model_keys_for_item(item):
    label = _normalize_display_key(item.get("normalized_label") or item.get("label") or "")

    if not label:
        return []

    if label in MEAT_SEGMENTATION_LABELS:
        return ["meat"]

    if label in SEAFOOD_SEGMENTATION_LABELS:
        return ["seafood"]

    if label in MUSHROOM_SEGMENTATION_LABELS:
        return ["mushroom"]

    if label in FRUIT_SEGMENTATION_LABELS:
        return ["fruit"]

    if label in VEGETABLE_SEGMENTATION_LABELS:
        return ["vegetable"]

    return []


def _not_ready_result(model_key, config):
    return {
        "model_key": model_key,
        "model_name": config["model_name"],
        "status": "not_ready",
        "label": "",
        "display_label": "未接入",
        "confidence": 0.0,
        "message": config["message"],
        "top_predictions": [],
    }


def _error_result(model_key, message):
    config = MODEL_CONFIGS[model_key]

    return {
        "model_key": model_key,
        "model_name": config["model_name"],
        "status": "error",
        "label": "",
        "display_label": "识别失败",
        "confidence": 0.0,
        "message": message,
        "top_predictions": [],
    }


def _predict_with_model(model_key, image_path: Path):
    if _IMPORT_ERROR:
        return _error_result(model_key, f"模型依赖导入失败：{_IMPORT_ERROR}")

    if not image_path.exists():
        return _error_result(model_key, f"切割图片不存在：{image_path}")

    try:
        model_bundle = _get_model(model_key)

        if model_key == "meat":
            return _predict_meat(model_bundle, image_path)

        return _predict_standard_classifier(model_key, model_bundle, image_path)
    except Exception as exc:
        return _error_result(model_key, str(exc))


def _get_model(model_key):
    if model_key in _loaded_models:
        return _loaded_models[model_key]

    if torch is None:
        raise RuntimeError("PyTorch 未安装，无法加载食材识别模型。")

    config = MODEL_CONFIGS[model_key]
    model_path = config["path"]

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _safe_torch_load(model_path, device)

    if model_key in {"fruit", "mushroom"}:
        bundle = _load_efficientnet_b0_model(model_key, checkpoint, device)
    elif model_key == "vegetable":
        bundle = _load_vegetable_model(checkpoint, device)
    elif model_key == "meat":
        bundle = _load_meat_model(checkpoint, device)
    elif model_key == "seafood":
        bundle = _load_seafood_model(checkpoint, device)
    else:
        raise ValueError(f"未支持的模型：{model_key}")

    _loaded_models[model_key] = bundle
    return bundle


def _safe_torch_load(model_path: Path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict"]:
            value = checkpoint.get(key)

            if isinstance(value, dict):
                return _clean_state_dict(value)

    return _clean_state_dict(checkpoint)


def _clean_state_dict(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        clean_key = str(key)

        for prefix in ["module.", "model."]:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]

        cleaned[clean_key] = value

    return cleaned


def _checkpoint_class_names(checkpoint):
    if not isinstance(checkpoint, dict):
        return None

    return checkpoint.get("class_names") or checkpoint.get("classes")


def _load_config_class_names(model_key):
    config = MODEL_CONFIGS[model_key]
    class_names_path = config.get("class_names_path")

    if class_names_path is None:
        return None

    if not class_names_path.exists():
        raise FileNotFoundError(f"类别文件不存在：{class_names_path}")

    with class_names_path.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"类别文件格式不正确：{class_names_path}")

    return [str(name) for name in class_names]


def _infer_out_features(state_dict, keys):
    for key in keys:
        value = state_dict.get(key)

        if value is not None:
            return int(value.shape[0])

    return None


def _load_efficientnet_b0_model(model_key, checkpoint, device):
    state_dict = _extract_state_dict(checkpoint)
    class_names = _checkpoint_class_names(checkpoint) or _load_config_class_names(model_key)
    out_features = _infer_out_features(state_dict, ["classifier.1.bias", "classifier.1.weight"])

    if not class_names:
        if out_features is None:
            raise ValueError(f"{model_key} 模型缺少类别名，且无法从权重推断输出层数量。")

        class_names = [f"{model_key}_class_{i}" for i in range(out_features)]

    if out_features is not None and out_features != len(class_names):
        raise ValueError(
            f"{model_key} 类别数量和模型输出层不一致："
            f"class_names={len(class_names)}，out_features={out_features}"
        )

    model = _new_efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "device": device,
        "class_names": list(class_names),
        "transform": _standard_transform(224),
    }


def _load_vegetable_model(checkpoint, device):
    state_dict = _extract_state_dict(checkpoint)
    class_names = _checkpoint_class_names(checkpoint)

    if not class_names:
        out_features = state_dict["fc.bias"].shape[0]
        class_names = [f"vegetable_class_{i}" for i in range(out_features)]

    model = _new_resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "device": device,
        "class_names": list(class_names),
        "transform": _standard_transform(224),
    }


def _load_seafood_model(checkpoint, device):
    state_dict = _extract_state_dict(checkpoint)
    class_names = _checkpoint_class_names(checkpoint)
    out_features = _infer_out_features(state_dict, ["fc.1.bias", "fc.1.weight"])

    if not class_names:
        if out_features is None:
            raise ValueError("seafood 模型缺少类别名，且无法从权重推断输出层数量。")

        class_names = [f"seafood_class_{i}" for i in range(out_features)]

    if out_features is not None and out_features != len(class_names):
        raise ValueError(
            "seafood 类别数量和模型输出层不一致："
            f"class_names={len(class_names)}，out_features={out_features}"
        )

    dropout = 0.0
    image_size = 224

    if isinstance(checkpoint, dict):
        dropout = float(checkpoint.get("dropout") or 0.0)
        image_size = int(checkpoint.get("image_size") or 224)

    model = _new_resnet50()
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.fc.in_features, len(class_names)),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "device": device,
        "class_names": list(class_names),
        "transform": _standard_transform(image_size),
    }


def _load_meat_model(checkpoint, device):
    state_dict = checkpoint["model_state_dict"]
    class_names = checkpoint.get("classes") or ["Chicken", "Pork", "Beef", "Other"]
    image_size = int(checkpoint.get("image_size", 224))

    model = _build_meat_model(len(class_names))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "device": device,
        "class_names": list(class_names),
        "transform": _standard_transform(image_size),
    }


def _new_resnet50():
    try:
        return models.resnet50(weights=None)
    except TypeError:
        return models.resnet50(pretrained=False)


def _new_efficientnet_b0():
    try:
        return models.efficientnet_b0(weights=None)
    except TypeError:
        return models.efficientnet_b0(pretrained=False)


def _standard_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _predict_standard_classifier(model_key, bundle, image_path: Path):
    image = Image.open(image_path).convert("RGB")
    tensor = bundle["transform"](image).unsqueeze(0).to(bundle["device"])

    with torch.no_grad():
        logits = bundle["model"](tensor)[0]
        probs = torch.softmax(logits, dim=0)

    return _format_prediction(model_key, probs, bundle["class_names"])


def _predict_meat(bundle, image_path: Path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    tensor = bundle["transform"](image).unsqueeze(0).to(bundle["device"])

    with torch.no_grad():
        logits = bundle["model"](tensor)[0]
        raw_probs = torch.softmax(logits, dim=0)
        adjusted_logits = logits.clone()

        classes = bundle["class_names"]
        marbling_score = _calc_marbling_score(image_np)
        color_score = _calc_beef_color_score(image_np)
        beef_aux_score = float(np.clip(0.65 * marbling_score + 0.35 * color_score, -1.0, 1.0))

        if "Pork" in classes and "Beef" in classes:
            pork_idx = classes.index("Pork")
            beef_idx = classes.index("Beef")
            pork_beef_prob = raw_probs[pork_idx].item() + raw_probs[beef_idx].item()

            if pork_beef_prob >= 0.35:
                adjusted_logits[beef_idx] += 0.8 * beef_aux_score
                adjusted_logits[pork_idx] -= 0.8 * beef_aux_score

        probs = torch.softmax(adjusted_logits, dim=0)

    result = _format_prediction("meat", probs, bundle["class_names"])

    if result["confidence"] < 0.70:
        result["label"] = "Unknown (非肉类或图片不清晰)"
        result["display_label"] = _display_label(result["label"])

    result["raw_top_predictions"] = _top_predictions(raw_probs, bundle["class_names"])
    result["aux_scores"] = {
        "marbling_score": round(marbling_score, 4),
        "color_score": round(color_score, 4),
        "beef_aux_score": round(beef_aux_score, 4),
    }

    return result


def _format_prediction(model_key, probs, class_names):
    config = MODEL_CONFIGS[model_key]
    confidence, index = torch.max(probs, dim=0)
    label = str(class_names[int(index.item())])
    confidence_value = round(float(confidence.item()), 4)

    return {
        "model_key": model_key,
        "model_name": config["model_name"],
        "status": "ok",
        "label": label,
        "display_label": _display_label(label),
        "confidence": confidence_value,
        "message": "",
        "top_predictions": _top_predictions(probs, class_names),
    }


def _top_predictions(probs, class_names, top_k=5):
    k = min(top_k, len(class_names))
    values, indices = torch.topk(probs, k=k)
    rows = []

    for value, index in zip(values, indices):
        label = str(class_names[int(index.item())])
        rows.append({
            "label": label,
            "display_label": _display_label(label),
            "confidence": round(float(value.item()), 4),
        })

    return rows


DISPLAY_LABEL_FIXED = {
    "unknown": "不明",
    "other": "その他",
    "no detection": "未検出",
    "unknown 非肉类或图片不清晰": "不明（肉類以外または画像不鮮明）",
    "chicken": "鶏肉",
    "pork": "豚肉",
    "beef": "牛肉",
}

DISPLAY_LABEL_FALLBACKS = {
    "clam": "ハマグリ",
    "cod": "タラ",
    "crab": "カニ",
    "eel": "ウナギ",
    "lobster": "ロブスター",
    "mackerel": "サバ",
    "octopus": "タコ",
    "oyster": "カキ",
    "salmon": "サケ",
    "scallop": "ホタテ",
    "shrimp": "エビ",
    "squid": "イカ",
    "tuna": "マグロ",
}


def _display_label(label: str) -> str:
    fallback = str(label or "").replace("_", " ").strip()
    normalized = _normalize_display_key(fallback)

    if normalized in DISPLAY_LABEL_FIXED:
        return DISPLAY_LABEL_FIXED[normalized]

    japanese_name = lookup_japanese_name([label, fallback])

    if japanese_name:
        return japanese_name

    if normalized in DISPLAY_LABEL_FALLBACKS:
        return DISPLAY_LABEL_FALLBACKS[normalized]

    return fallback


def _normalize_display_key(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = text.replace("(", " ").replace(")", " ")
    text = re.sub(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class GaborTextureExtractor(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.kernels = nn.Parameter(self._make_gabor_kernels(), requires_grad=False)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, out_dim),
            nn.ReLU(inplace=True),
        )

    def _make_gabor_kernels(self):
        import math

        kernels = []
        ksize = 15
        sigma = 3.0

        for theta in [0, 45, 90, 135]:
            for freq in [0.1, 0.25]:
                theta_rad = theta * math.pi / 180
                kernel = np.zeros((ksize, ksize), dtype=np.float32)
                half = ksize // 2

                for y in range(-half, half + 1):
                    for x in range(-half, half + 1):
                        x_rot = x * math.cos(theta_rad) + y * math.sin(theta_rad)
                        y_rot = -x * math.sin(theta_rad) + y * math.cos(theta_rad)
                        gauss = math.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
                        sinusoid = math.cos(2 * math.pi * freq * x_rot)
                        kernel[y + half, x + half] = gauss * sinusoid

                kernel -= kernel.mean()
                std = kernel.std()

                if std > 0:
                    kernel /= std

                kernels.append(kernel)

        return torch.from_numpy(np.stack(kernels)[:, np.newaxis, :, :])

    def forward(self, x):
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        filtered = F.conv2d(gray, self.kernels, padding=7)
        filtered = torch.abs(filtered)
        pooled = self.pool(filtered)
        return self.fc(pooled)


def _build_meat_model(num_classes: int):
    backbone = _new_resnet50()
    backbone.fc = nn.Identity()

    class MeatClassifierWithTexture(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.gabor = GaborTextureExtractor(out_dim=64)
            self.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(2048 + 64, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            feat_resnet = self.backbone(x)
            feat_texture = self.gabor(x)
            combined = torch.cat([feat_resnet, feat_texture], dim=1)
            return self.classifier(combined)

    return MeatClassifierWithTexture()


def _calc_marbling_score(img_np) -> float:
    if cv2 is None:
        return 0.0

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    meat_mask = ((h <= 20) | (h >= 155)) & (s > 40) & (v > 60)

    if meat_mask.sum() < 500:
        return 0.0

    fat_mask = (s < 50) & (v > 160)
    fat_in_meat = fat_mask & meat_mask
    fat_density = fat_in_meat.sum() / max(meat_mask.sum(), 1)

    h_img, w_img = img_np.shape[:2]
    grid_densities = []

    for gy in range(4):
        for gx in range(4):
            y1 = gy * h_img // 4
            y2 = (gy + 1) * h_img // 4
            x1 = gx * w_img // 4
            x2 = (gx + 1) * w_img // 4
            cell_meat = meat_mask[y1:y2, x1:x2].sum()
            cell_fat = fat_in_meat[y1:y2, x1:x2].sum()

            if cell_meat > 50:
                grid_densities.append(cell_fat / cell_meat)

    if not grid_densities:
        return 0.0

    uniformity = 1.0 - float(np.std(grid_densities))

    if fat_density > 0.15:
        score = min(1.0, fat_density * 3.0) * max(0.0, uniformity)
    elif fat_density > 0.08:
        score = fat_density * 1.5 * max(0.0, uniformity)
    else:
        score = -0.1

    return float(np.clip(score, -1.0, 1.0))


def _calc_beef_color_score(img_np) -> float:
    if cv2 is None:
        return 0.0

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    meat_mask = ((h <= 20) | (h >= 155)) & (s > 40) & (v > 60)

    if meat_mask.sum() < 500:
        return 0.0

    mean_s = float(s[meat_mask].mean())
    mean_v = float(v[meat_mask].mean())
    color_score = (mean_s - 90) / 60.0
    brightness_penalty = (mean_v - 150) / 100.0
    color_score -= brightness_penalty * 0.3

    return float(np.clip(color_score, -1.0, 1.0))
