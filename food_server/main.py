from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
from threading import Lock
import uuid

from modules.food_classifier import classify_cut_items
from modules.image_cutter import cut_food_image
from modules.nutrition_analyzer import analyze_nutrition


# =========================
# 1. 基础路径
# =========================
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
CROP_OUTPUT_DIR = OUTPUT_DIR / "crops"
DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CROP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

ANALYSIS_TASKS = {}
ANALYSIS_TASK_LOCK = Lock()


# =========================
# 2. FastAPI
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 让手机 App 可以访问 outputs 里的图片
# 例如：http://192.168.40.90:8000/outputs/crops/xxx.jpg
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


# =========================
# 3. 上传限制
# =========================
MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "application/octet-stream",
}


# =========================
# 4. 工具函数
# =========================
def clear_output_dir():
    """
    每次分析前清空 outputs 文件夹。
    这样 outputs/crops 不会越积越多。
    """
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    OUTPUT_DIR.mkdir(exist_ok=True)
    CROP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_crop_urls(cut_items):
    """
    给每个切割结果加上手机可以访问的图片 URL。
    """
    new_items = []

    for item in cut_items:
        crop_name = item.get("crop_name", "")

        new_item = dict(item)
        new_item["crop_url"] = f"/outputs/crops/{crop_name}"

        new_items.append(new_item)

    return new_items


def set_analysis_task(task_id, **updates):
    with ANALYSIS_TASK_LOCK:
        task = ANALYSIS_TASKS.setdefault(task_id, {})
        task.update(updates)
        return dict(task)


def get_analysis_task(task_id):
    with ANALYSIS_TASK_LOCK:
        task = ANALYSIS_TASKS.get(task_id)
        return dict(task) if task is not None else None


def run_analysis_task(task_id, file_bytes, original_filename, content_type):
    try:
        result = run_analysis_pipeline(
            file_bytes=file_bytes,
            original_filename=original_filename,
            content_type=content_type,
            progress=lambda stage: set_analysis_task(
                task_id,
                status="running",
                stage=stage,
            ),
        )
        set_analysis_task(
            task_id,
            status="complete",
            stage="分析完了",
            result=result,
            error="",
        )
    except HTTPException as exc:
        set_analysis_task(
            task_id,
            status="error",
            stage="分析失敗",
            error=str(exc.detail),
        )
    except Exception as exc:
        set_analysis_task(
            task_id,
            status="error",
            stage="分析失敗",
            error=str(exc),
        )


def run_analysis_pipeline(file_bytes, original_filename, content_type, progress=None):
    progress = progress or (lambda _stage: None)
    suffix = Path(original_filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"只允许上传 jpg、jpeg、png 图片。"
                f"当前文件名：{original_filename}，类型：{content_type}"
            )
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="图片太大，最大只能上传 10MB"
        )

    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".jpg"

    filename = f"{uuid.uuid4().hex}{suffix}"
    image_path = UPLOAD_DIR / filename

    progress("画像を保存中...")

    with open(image_path, "wb") as f:
        f.write(file_bytes)

    clear_output_dir()

    progress("画像切り出し中...")
    cut_result = cut_food_image(
        image_path=image_path,
        output_dir=CROP_OUTPUT_DIR
    )

    cut_items = add_crop_urls(cut_result["items"])
    cut_count = len(cut_items)

    progress(f"食材認識中... 切り出し {cut_count} 枚")
    recognition_result = classify_cut_items(cut_items)
    cut_items = recognition_result["items"]
    recognition_time = recognition_result["time"]

    progress("栄養成分分析中...")
    nutrition_result = analyze_nutrition(cut_items, OUTPUT_DIR)
    cut_items = nutrition_result["items"]
    nutrition_results = nutrition_result["nutrition_results"]
    fat_lean_results = nutrition_result["fat_lean_results"]
    nutrition_time = nutrition_result["time"]
    total_time = round(cut_result["time"] + recognition_time + nutrition_time, 3)

    progress("結果を整理中...")
    best_food = None

    for item in cut_items:
        best = item.get("best_recognition")

        if not best:
            continue

        if best_food is None or best.get("confidence", 0.0) > best_food.get("confidence", 0.0):
            best_food = best

    food_name = best_food.get("display_label", "未识别") if best_food else "未识别"
    confidence = best_food.get("confidence", 0.0) if best_food else 0.0

    ok_fat_lean_count = len([r for r in fat_lean_results if r.get("status") == "ok"])
    ok_nutrition_count = len([r for r in nutrition_results if r.get("status") == "ok"])

    if fat_lean_results:
        meat_detail = (
            f"检测到 {len(fat_lean_results)} 张牛肉/猪肉切割图，"
            f"其中 {ok_fat_lean_count} 张完成脂肪/瘦肉比例分析。"
        )
    else:
        meat_detail = "本次没有识别出需要脂肪/瘦肉比例分析的牛肉或猪肉。"

    steps = [
        {
            "title": "图片切割",
            "detail": f"从原图中切割出 {cut_count} 个候选食材区域。",
            "time": cut_result["time"],
        },
        {
            "title": "食材识别",
            "detail": (
                f"已对 {cut_count} 张切割图执行水果、蔬菜、蘑菇、肉类和海鲜模型识别。"
            ),
            "time": recognition_time,
        },
        {
            "title": "肉类脂肪分析",
            "detail": meat_detail,
            "time": nutrition_time,
        },
        {
            "title": "营养成分分析",
            "detail": (
                f"已为 {ok_nutrition_count} 个识别食材生成营养成分结果。"
                "普通食材查询 NutritionFacts.xlsx；牛肉/猪肉按脂肪/瘦肉比例估算。"
            ),
            "time": nutrition_time,
        },
    ]

    result_text = (
        "图片切割、食材识别和营养成分分析完成。\n"
        f"切割区域数量：{cut_count} 个\n"
        f"当前最高置信度识别：{food_name} ({confidence * 100:.1f}%)\n"
        f"营养结果数量：{ok_nutrition_count} 个\n"
        f"肉类脂肪分析数量：{ok_fat_lean_count} 个"
    )

    return {
        "ok": True,
        "filename": filename,
        "result_text": result_text,
        "food_name": food_name,
        "confidence": confidence,
        "total_time": total_time,
        "cut_count": cut_count,
        "cut_items": cut_items,
        "nutrition_results": nutrition_results,
        "fat_lean_results": fat_lean_results,
        "steps": steps,
    }


# =========================
# 5. 接口
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Food server is running"
    }


@app.post("/analyze/start")
async def analyze_start(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    file_bytes = await file.read()
    task_id = uuid.uuid4().hex
    set_analysis_task(
        task_id,
        status="queued",
        stage="分析待機中...",
        result=None,
        error="",
    )
    background_tasks.add_task(
        run_analysis_task,
        task_id,
        file_bytes,
        file.filename or "",
        file.content_type,
    )

    return {
        "ok": True,
        "task_id": task_id,
        "status": "queued",
        "stage": "分析待機中...",
    }


@app.get("/analyze/status/{task_id}")
def analyze_status(task_id: str):
    task = get_analysis_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail="分析任务不存在。")

    return {
        "ok": True,
        "task_id": task_id,
        **task,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    手机 App 上传图片后进入这里。

    当前流程：
    1. 接收图片
    2. 保存原图
    3. 清空旧 outputs
    4. 执行图片切割
    5. 食材识别
    6. 肉类脂肪/瘦肉比例分析与营养成分分析
    7. 返回切割图片 URL 和流程结果
    """

    file_bytes = await file.read()

    return run_analysis_pipeline(
        file_bytes=file_bytes,
        original_filename=file.filename or "",
        content_type=file.content_type,
    )
