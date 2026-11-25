from __future__ import annotations

import argparse
import re
import time
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

MODULE_DIR = Path(__file__).resolve().parent
NUTRITION_FILE = MODULE_DIR / "NutritionFacts.xlsx"

try:
    from openpyxl import load_workbook

    _OPENPYXL_ERROR = None
except Exception as exc:  # pragma: no cover - depends on server environment
    load_workbook = None
    _OPENPYXL_ERROR = exc

try:
    from modules.fat_lean_ratio import analyze_one_image

    _FAT_LEAN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on server environment
    analyze_one_image = None
    _FAT_LEAN_IMPORT_ERROR = exc


NUTRIENT_COLUMNS = {
    "energy_kcal": "エネルギー",
    "protein_g": "たんぱく質",
    "fat_g": "脂質",
    "carbohydrate_g": "炭水化物",
}

GENERIC_ALIASES = {
    "bok choy": "Wuta-tsai",
    "bokchoy": "Wuta-tsai",
    "bell pepper": "Pepper",
    "bell peppers": "Pepper",
    "capsicum": "Pepper",
    "chicken": "Chicken breast",
    "pork": "Pork legs",
    "beef": "Beef shank",
}

MEAT_NUTRITION_BASES = {
    "pork": {
        "display_name": "Pork",
        "lean": "Pork legs",
        "fat": "Pork belly",
    },
    "beef": {
        "display_name": "Beef",
        "lean": "Beef shank",
        "fat": "Beef ribs",
    },
}

NEGATIVE_LABELS = {
    "",
    "other",
    "unknown",
    "unknown non meat or unclear image",
    "unknown 非肉类或图片不清晰",
    "no detection",
    "未识别",
    "未接入",
    "识别失败",
}

_nutrition_records: list[dict[str, Any]] | None = None
_nutrition_index: dict[str, dict[str, Any]] | None = None
_XLSX_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def lookup_japanese_name(candidates: list[str] | tuple[str, ...]) -> str:
    try:
        record, _ = _find_nutrition_record([str(value) for value in candidates])
    except Exception:
        return ""

    if record is None:
        return ""

    return str(record.get("japanese_name") or "").strip()


def analyze_nutrition(cut_items: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    start_time = time.perf_counter()
    updated_items = []
    nutrition_results = []
    fat_lean_results = []

    for index, item in enumerate(cut_items, start=1):
        updated_item = dict(item)
        food_name = _best_food_name(updated_item)
        meat_type = _meat_type(updated_item)

        fat_lean_result = None

        if meat_type in MEAT_NUTRITION_BASES:
            fat_lean_result = _analyze_fat_lean(index, updated_item, meat_type, output_dir)
            updated_item["fat_lean_result"] = fat_lean_result
            fat_lean_results.append(fat_lean_result)

        if not _is_recognized_food(updated_item):
            nutrition_result = _base_nutrition_result(
                index=index,
                item=updated_item,
                food_name=food_name,
                status="skipped",
                message="没有可用于营养查询的识别结果。",
            )
        elif meat_type in MEAT_NUTRITION_BASES and fat_lean_result and fat_lean_result["status"] == "ok":
            nutrition_result = _calculate_meat_nutrition(
                index=index,
                item=updated_item,
                meat_type=meat_type,
                fat_lean_result=fat_lean_result,
            )
        else:
            nutrition_result = _lookup_nutrition_result(
                index=index,
                item=updated_item,
                food_name=food_name,
                extra_candidates=[meat_type] if meat_type else [],
            )

        updated_item["nutrition_result"] = nutrition_result

        if nutrition_result["status"] != "skipped":
            nutrition_results.append(nutrition_result)

        updated_items.append(updated_item)

    elapsed = round(time.perf_counter() - start_time, 3)

    return {
        "items": updated_items,
        "nutrition_results": nutrition_results,
        "fat_lean_results": fat_lean_results,
        "time": elapsed,
    }


def _load_nutrition_table() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    global _nutrition_records, _nutrition_index

    if _nutrition_records is not None and _nutrition_index is not None:
        return _nutrition_records, _nutrition_index

    if not NUTRITION_FILE.exists():
        raise FileNotFoundError(f"营养表不存在：{NUTRITION_FILE}")

    records = []
    index = {}

    for row in _read_nutrition_rows(NUTRITION_FILE):
        english_name = str(row.get("EnglishName") or "").strip()

        if not english_name:
            continue

        record = {
            "id": row.get("ID"),
            "english_name": english_name,
            "japanese_name": str(row.get("日本語名") or "").strip(),
            "matched_food_name": str(row.get("対応食品名") or "").strip(),
            "source_name": str(row.get("データ源") or "").strip(),
            "source_url": str(row.get("ソースURL") or "").strip(),
            "completion_method": str(row.get("補完方法") or "").strip(),
            "note": str(row.get("信頼度・備考") or "").strip(),
        }

        for field, column in NUTRIENT_COLUMNS.items():
            record[field] = _clean_nutrient(row.get(column))

        search_text = " ".join(
            [
                english_name,
                record["japanese_name"],
                record["matched_food_name"],
            ]
        )
        record["search_key"] = _normalize_key(search_text)

        records.append(record)
        index[_normalize_key(english_name)] = record

    _nutrition_records = records
    _nutrition_index = index
    return records, index


def _read_nutrition_rows(path: Path) -> list[dict[str, Any]]:
    if _OPENPYXL_ERROR is None and load_workbook is not None:
        workbook = load_workbook(path, data_only=True, read_only=True)
        sheet = workbook[workbook.sheetnames[0]]
        headers = [
            sheet.cell(1, col).value for col in range(1, sheet.max_column + 1)
        ]
        return [
            dict(zip(headers, values))
            for values in sheet.iter_rows(min_row=2, values_only=True)
        ]

    return _read_xlsx_rows_without_openpyxl(path)


def _read_xlsx_rows_without_openpyxl(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _read_shared_strings(archive)
        sheet_path = _first_sheet_path(archive)
        root = ET.fromstring(archive.read(sheet_path))

    sheet_rows = root.findall(f".//{_XLSX_MAIN_NS}sheetData/{_XLSX_MAIN_NS}row")

    if not sheet_rows:
        return []

    header_cells = _read_xlsx_row(sheet_rows[0], shared_strings)
    max_header_index = max(header_cells.keys(), default=-1)
    headers = [
        str(header_cells.get(index) or "").strip()
        for index in range(max_header_index + 1)
    ]
    rows = []

    for sheet_row in sheet_rows[1:]:
        cells = _read_xlsx_row(sheet_row, shared_strings)
        row = {}

        for index, header in enumerate(headers):
            if header:
                row[header] = cells.get(index)

        rows.append(row)

    return rows


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []

    for item in root.findall(f"{_XLSX_MAIN_NS}si"):
        parts = [
            node.text or ""
            for node in item.findall(f".//{_XLSX_MAIN_NS}t")
        ]
        strings.append("".join(parts))

    return strings


def _first_sheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    first_sheet = workbook.find(f"{_XLSX_MAIN_NS}sheets/{_XLSX_MAIN_NS}sheet")
    relationship_id = first_sheet.attrib.get(
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id",
        "",
    ) if first_sheet is not None else ""

    if relationship_id and "xl/_rels/workbook.xml.rels" in archive.namelist():
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))

        for relationship in relationships:
            if relationship.attrib.get("Id") == relationship_id:
                target = relationship.attrib.get("Target", "")

                if target.startswith("/"):
                    return target.lstrip("/")

                return f"xl/{target}".replace("\\", "/")

    worksheet_names = [
        name for name in archive.namelist()
        if name.startswith("xl/worksheets/") and name.endswith(".xml")
    ]

    if not worksheet_names:
        raise FileNotFoundError("xlsx 文件中没有 worksheet。")

    return sorted(worksheet_names)[0]


def _read_xlsx_row(row: ET.Element, shared_strings: list[str]) -> dict[int, Any]:
    values = {}

    for cell in row.findall(f"{_XLSX_MAIN_NS}c"):
        cell_ref = cell.attrib.get("r", "")
        column_index = _column_index_from_cell_ref(cell_ref)

        if column_index < 0:
            continue

        values[column_index] = _read_xlsx_cell(cell, shared_strings)

    return values


def _read_xlsx_cell(cell: ET.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t", "")

    if cell_type == "inlineStr":
        return "".join(
            node.text or ""
            for node in cell.findall(f".//{_XLSX_MAIN_NS}t")
        )

    value_node = cell.find(f"{_XLSX_MAIN_NS}v")
    raw_value = value_node.text if value_node is not None else ""

    if cell_type == "s":
        index = int(raw_value or 0)
        return shared_strings[index] if 0 <= index < len(shared_strings) else ""

    if cell_type == "b":
        return raw_value == "1"

    return _parse_xlsx_scalar(raw_value)


def _column_index_from_cell_ref(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Za-z]", "", cell_ref)

    if not letters:
        return -1

    index = 0

    for char in letters.upper():
        index = index * 26 + (ord(char) - ord("A") + 1)

    return index - 1


def _parse_xlsx_scalar(value: Any) -> Any:
    text = str(value or "").strip()

    if not text:
        return ""

    try:
        number = float(text)
    except ValueError:
        return text

    if number.is_integer():
        return int(number)

    return number


def _lookup_nutrition_result(
    *,
    index: int,
    item: dict[str, Any],
    food_name: str,
    extra_candidates: list[str] | None = None,
) -> dict[str, Any]:
    base_result = _base_nutrition_result(
        index=index,
        item=item,
        food_name=food_name,
        status="not_found",
        message="营养表中没有找到对应食材。",
    )

    try:
        record, match_method = _find_nutrition_record(
            _candidate_names(item, food_name, extra_candidates or [])
        )
    except Exception as exc:
        base_result["status"] = "error"
        base_result["message"] = str(exc)
        return base_result

    if record is None:
        return base_result

    return _record_to_nutrition_result(
        base_result=base_result,
        record=record,
        basis="nutrition_table",
        match_method=match_method,
        message="已从 NutritionFacts.xlsx 查询营养成分。",
    )


def _calculate_meat_nutrition(
    *,
    index: int,
    item: dict[str, Any],
    meat_type: str,
    fat_lean_result: dict[str, Any],
) -> dict[str, Any]:
    config = MEAT_NUTRITION_BASES[meat_type]
    base_result = _base_nutrition_result(
        index=index,
        item=item,
        food_name=config["display_name"],
        status="error",
        message="肉类营养估算失败。",
    )

    try:
        _, nutrition_index = _load_nutrition_table()
        lean_record = nutrition_index.get(_normalize_key(config["lean"]))
        fat_record = nutrition_index.get(_normalize_key(config["fat"]))
    except Exception as exc:
        base_result["message"] = str(exc)
        return base_result

    if lean_record is None or fat_record is None:
        base_result["status"] = "not_found"
        base_result["message"] = "营养表中缺少肉类脂肪/瘦肉估算所需的基准数据。"
        return base_result

    fat_ratio = _safe_float(fat_lean_result.get("fat_ratio"))
    lean_ratio = _safe_float(fat_lean_result.get("lean_ratio"))
    ratio_total = fat_ratio + lean_ratio

    if ratio_total <= 0:
        lean_ratio = 1.0
        fat_ratio = 0.0
        ratio_total = 1.0

    fat_weight = fat_ratio / ratio_total
    lean_weight = lean_ratio / ratio_total

    base_result.update(
        {
            "status": "ok",
            "message": "已按脂肪/瘦肉比例估算营养成分。",
            "basis": "fat_lean_ratio",
            "unit": "每100g估算",
            "match_name": f"{lean_record['english_name']} + {fat_record['english_name']}",
            "match_method": "fat_lean_weighted",
            "source_name": _join_unique([lean_record["source_name"], fat_record["source_name"]]),
            "source_url": _join_unique([lean_record["source_url"], fat_record["source_url"]]),
            "fat_lean_result": fat_lean_result,
        }
    )

    for field in NUTRIENT_COLUMNS:
        value = lean_record[field] * lean_weight + fat_record[field] * fat_weight
        base_result[field] = round(max(value, 0.0), 2)

    return base_result


def _find_nutrition_record(candidates: list[str]) -> tuple[dict[str, Any] | None, str]:
    records, nutrition_index = _load_nutrition_table()

    normalized_candidates = []

    for candidate in candidates:
        normalized = _normalize_key(candidate)

        if normalized and normalized not in normalized_candidates:
            normalized_candidates.append(normalized)

        alias = GENERIC_ALIASES.get(normalized)

        if alias:
            alias_key = _normalize_key(alias)

            if alias_key not in normalized_candidates:
                normalized_candidates.append(alias_key)

    for normalized in normalized_candidates:
        record = nutrition_index.get(normalized)

        if record is not None:
            return record, "exact"

    for normalized in normalized_candidates:
        for record in records:
            if normalized and normalized in record["search_key"]:
                return record, "contains"

    return None, ""


def _analyze_fat_lean(
    index: int,
    item: dict[str, Any],
    meat_type: str,
    output_dir: Path,
) -> dict[str, Any]:
    crop_name = str(item.get("crop_name") or "")
    food_name = _best_food_name(item)
    crop_path = output_dir / "crops" / crop_name

    result = {
        "item_index": index,
        "crop_name": crop_name,
        "food_name": food_name,
        "meat_type": meat_type,
        "status": "error",
        "message": "",
        "fat_ratio": 0.0,
        "lean_ratio": 0.0,
        "unknown_ratio": 0.0,
        "fat_pixels": 0,
        "lean_pixels": 0,
        "unknown_pixels": 0,
        "overlay_url": "",
        "visualization_url": "",
    }

    if _FAT_LEAN_IMPORT_ERROR is not None or analyze_one_image is None:
        result["message"] = f"脂肪/瘦肉分析模块导入失败：{_FAT_LEAN_IMPORT_ERROR}"
        return result

    if not crop_path.exists():
        result["message"] = f"切割图片不存在：{crop_path}"
        return result

    try:
        output_root = output_dir / "fat_lean" / crop_path.stem
        summary, _ = analyze_one_image(
            image_path=crop_path,
            args=_fat_lean_args(),
            output_root=output_root,
        )
    except Exception as exc:
        result["message"] = str(exc)
        return result

    status = str(summary.get("status") or "error")
    stem = crop_path.stem
    overlay_path = output_root / "overlays" / f"{stem}_overlay.jpg"
    visualization_path = output_root / "visualizations" / f"{stem}_fat_lean_visual.png"

    result.update(
        {
            "status": status,
            "message": "脂肪/瘦肉比例分析完成。" if status == "ok" else f"脂肪/瘦肉比例分析未完成：{status}",
            "fat_ratio": round(_safe_float(summary.get("fat_ratio_excluding_unknown")), 4),
            "lean_ratio": round(_safe_float(summary.get("lean_ratio_excluding_unknown")), 4),
            "unknown_ratio": round(_safe_float(summary.get("unknown_ratio_inside_meat")), 4),
            "fat_pixels": int(_safe_float(summary.get("fat_pixels"))),
            "lean_pixels": int(_safe_float(summary.get("lean_pixels"))),
            "unknown_pixels": int(_safe_float(summary.get("unknown_pixels"))),
            "overlay_url": _output_url(output_dir, overlay_path),
            "visualization_url": _output_url(output_dir, visualization_path),
        }
    )

    return result


def _base_nutrition_result(
    *,
    index: int,
    item: dict[str, Any],
    food_name: str,
    status: str,
    message: str,
) -> dict[str, Any]:
    best = item.get("best_recognition") or {}

    return {
        "item_index": index,
        "crop_name": str(item.get("crop_name") or ""),
        "food_name": food_name or "未识别",
        "confidence": _safe_float(item.get("food_confidence") or best.get("confidence")),
        "model_key": str(item.get("food_model_key") or best.get("model_key") or ""),
        "model_name": str(item.get("food_model_name") or best.get("model_name") or ""),
        "status": status,
        "message": message,
        "basis": "",
        "unit": "每100g",
        "match_name": "",
        "match_method": "",
        "source_name": "",
        "source_url": "",
        "energy_kcal": 0.0,
        "protein_g": 0.0,
        "fat_g": 0.0,
        "carbohydrate_g": 0.0,
    }


def _record_to_nutrition_result(
    *,
    base_result: dict[str, Any],
    record: dict[str, Any],
    basis: str,
    match_method: str,
    message: str,
) -> dict[str, Any]:
    base_result.update(
        {
            "status": "ok",
            "message": message,
            "basis": basis,
            "unit": "每100g",
            "match_name": record["english_name"],
            "match_method": match_method,
            "japanese_name": record["japanese_name"],
            "matched_food_name": record["matched_food_name"],
            "source_name": record["source_name"],
            "source_url": record["source_url"],
            "completion_method": record["completion_method"],
            "note": record["note"],
        }
    )

    for field in NUTRIENT_COLUMNS:
        base_result[field] = round(record[field], 2)

    return base_result


def _candidate_names(
    item: dict[str, Any],
    food_name: str,
    extra_candidates: list[str],
) -> list[str]:
    best = item.get("best_recognition") or {}
    candidates = [
        food_name,
        item.get("food_name"),
        best.get("display_label"),
        best.get("label"),
        item.get("normalized_label"),
        item.get("label"),
        *extra_candidates,
    ]

    return [str(candidate).strip() for candidate in candidates if str(candidate or "").strip()]


def _best_food_name(item: dict[str, Any]) -> str:
    best = item.get("best_recognition") or {}

    for value in [
        item.get("food_name"),
        best.get("display_label"),
        best.get("label"),
    ]:
        text = str(value or "").strip()

        if text and not _is_negative_text(text):
            return text

    return "未识别"


def _is_recognized_food(item: dict[str, Any]) -> bool:
    best = item.get("best_recognition") or {}

    for value in [
        item.get("food_name"),
        best.get("display_label"),
        best.get("label"),
    ]:
        text = str(value or "").strip()

        if text and not _is_negative_text(text):
            return True

    return False


def _meat_type(item: dict[str, Any]) -> str:
    best = item.get("best_recognition") or {}

    for value in [
        item.get("food_name"),
        best.get("display_label"),
        best.get("label"),
    ]:
        normalized = _normalize_key(value)

        if normalized == "pork" or normalized.startswith("pork "):
            return "pork"

        if normalized == "beef" or normalized.startswith("beef "):
            return "beef"

    return ""


def _is_negative_text(value: str) -> bool:
    normalized = _normalize_key(value)
    return normalized in NEGATIVE_LABELS


def _normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = text.replace("(", " ").replace(")", " ")
    text = re.sub(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_nutrient(value: Any) -> float:
    return max(_safe_float(value), 0.0)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _join_unique(values: list[str]) -> str:
    unique_values = []

    for value in values:
        text = str(value or "").strip()

        if text and text not in unique_values:
            unique_values.append(text)

    return " / ".join(unique_values)


def _output_url(output_dir: Path, path: Path) -> str:
    if not path.exists():
        return ""

    relative = path.relative_to(output_dir).as_posix()
    return f"/outputs/{relative}"


def _fat_lean_args() -> argparse.Namespace:
    return argparse.Namespace(
        clusters=5,
        alpha_threshold=10,
        erode=0,
        max_sample_pixels=200000,
        seed=42,
        fat_margin=0.08,
        lean_tolerance=0.05,
        min_fat_l=128.0,
        max_fat_s=150.0,
        max_fat_a=45.0,
        force_lean_a=24.0,
        force_lean_s=45.0,
        min_valid_v=25.0,
        min_valid_l=30.0,
    )
