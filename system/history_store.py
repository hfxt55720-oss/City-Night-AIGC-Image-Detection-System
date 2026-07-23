import json
from datetime import datetime
from pathlib import Path

from system.app_config import path_for


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "data" / "history.json"


def current_history_path() -> Path:
    return path_for("history")


def read_history(path: Path | None = None) -> list[dict]:
    target_path = Path(path) if path is not None else current_history_path()
    if not target_path.exists():
        return []

    try:
        with target_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        return []

    return data if isinstance(data, list) else []


def write_history(records: list[dict], path: Path | None = None):
    target_path = Path(path) if path is not None else current_history_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def append_history(record: dict, path: Path | None = None) -> dict:
    records = read_history(path)
    normalized = normalize_record(record)
    records.append(normalized)
    write_history(records, path)
    return normalized


def append_many_history(new_records: list[dict], path: Path | None = None) -> list[dict]:
    if not new_records:
        return []

    records = read_history(path)
    normalized_records = [normalize_record(record) for record in new_records]
    records.extend(normalized_records)
    write_history(records, path)
    return normalized_records


def normalize_record(record: dict) -> dict:
    ai_probability = to_float(record.get("ai_probability"))
    real_probability = to_float(record.get("real_probability"))
    confidence = max(ai_probability, real_probability)
    result = normalize_result(record.get("result", ""))
    timestamp = record.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gradcam = normalize_gradcam(record.get("gradcam"))

    normalized = {
        "id": record.get("id") or datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": timestamp,
        "source": record.get("source", "single"),
        "image_name": record.get("image_name", ""),
        "image_path": record.get("image_path", ""),
        "model_key": record.get("model_key", ""),
        "model_name": record.get("model_name", ""),
        "result": result,
        "ai_probability": round(ai_probability, 6),
        "real_probability": round(real_probability, 6),
        "confidence": round(confidence, 6),
        "time": record.get("time", ""),
    }
    if record.get("saved_at"):
        normalized["saved_at"] = str(record.get("saved_at"))
    if record.get("result_file"):
        normalized["result_file"] = str(record.get("result_file"))
    if record.get("device"):
        normalized["device"] = str(record.get("device"))
    if gradcam:
        normalized["gradcam"] = gradcam
    return normalized


def normalize_gradcam(value) -> dict:
    if not isinstance(value, dict):
        return {}

    keys = [
        "original_path",
        "heatmap_path",
        "overlay_path",
        "target_class",
        "target_layer",
        "model_name",
        "checkpoint_path",
    ]
    return {key: str(value[key]) for key in keys if value.get(key) not in (None, "")}


def summarize_history(records: list[dict]) -> dict:
    total = len(records)
    aigc_count = sum(1 for record in records if normalize_result(record.get("result")) == "AIGC")
    real_count = sum(1 for record in records if normalize_result(record.get("result")) == "REAL")
    confidences = [to_float(record.get("confidence", max_probability(record))) for record in records]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    model_counts = {}
    for record in records:
        model_name = record.get("model_name") or "未知模型"
        model_counts[model_name] = model_counts.get(model_name, 0) + 1

    return {
        "total": total,
        "aigc_count": aigc_count,
        "real_count": real_count,
        "avg_confidence": avg_confidence,
        "model_counts": model_counts,
    }


def max_probability(record: dict) -> float:
    return max(to_float(record.get("ai_probability")), to_float(record.get("real_probability")))


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_result(value) -> str:
    text = str(value).strip().upper()
    if text in {"AIGC", "AI", "FAKE", "AI生成", "生成图像"}:
        return "AIGC"
    if text in {"REAL", "REAL_IMAGE", "真实", "真实图像", "真实图片"}:
        return "REAL"
    return text or "UNKNOWN"
