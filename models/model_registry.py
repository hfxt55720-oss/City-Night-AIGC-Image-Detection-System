import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from system.app_config import APP_ROOT, resource_path


PACKAGED_WEIGHT_DIR = resource_path("models/weights")
LOCAL_WEIGHT_DIR = APP_ROOT / "models" / "weights"
WEIGHT_DIR = PACKAGED_WEIGHT_DIR
CUSTOM_WEIGHT_DIR = LOCAL_WEIGHT_DIR / "custom"
CUSTOM_MODELS_PATH = APP_ROOT / "config" / "custom_models.json"


SUPPORTED_MODEL_TYPES = {
    "aide": "AIDE-Night",
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
}


BASE_MODEL_SPECS = {
    "aide": {
        "name": "AIDE-Night",
        "architecture": "aide",
        "weight_file": "aide_night_best.pth",
        "weight_path": WEIGHT_DIR / "aide_night_best.pth",
        "inference_ready": True,
        "custom": False,
    },
    "resnet50": {
        "name": "ResNet50",
        "architecture": "resnet50",
        "weight_file": "resnet50_ai_detector.pth",
        "weight_path": WEIGHT_DIR / "resnet50_ai_detector.pth",
        "inference_ready": True,
        "custom": False,
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "architecture": "efficientnet_b0",
        "weight_file": "efficientnet_b0_ai_detector.pth",
        "weight_path": WEIGHT_DIR / "efficientnet_b0_ai_detector.pth",
        "inference_ready": True,
        "custom": False,
    },
}


MODEL_SPECS: dict[str, dict] = {}


def reload_model_specs() -> dict[str, dict]:
    MODEL_SPECS.clear()
    MODEL_SPECS.update({key: dict(value) for key, value in BASE_MODEL_SPECS.items()})
    for entry in load_custom_model_entries():
        spec = custom_entry_to_spec(entry)
        if spec is not None:
            MODEL_SPECS[entry["key"]] = spec
    return MODEL_SPECS


def load_custom_model_entries() -> list[dict]:
    if not CUSTOM_MODELS_PATH.exists():
        return []
    try:
        with CUSTOM_MODELS_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return []

    models = data.get("models", data if isinstance(data, list) else [])
    return models if isinstance(models, list) else []


def save_custom_model_entries(entries: list[dict]):
    CUSTOM_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CUSTOM_MODELS_PATH.open("w", encoding="utf-8") as file:
        json.dump({"models": entries}, file, ensure_ascii=False, indent=2)


def custom_entry_to_spec(entry: dict) -> dict | None:
    key = str(entry.get("key", "")).strip()
    architecture = str(entry.get("architecture", "")).strip()
    name = str(entry.get("name", "")).strip()
    weight_file = str(entry.get("weight_file", "")).strip()
    if not key or architecture not in SUPPORTED_MODEL_TYPES or not name or not weight_file:
        return None

    weight_path = _resolve_weight_file(weight_file)
    return {
        "name": name,
        "architecture": architecture,
        "weight_file": weight_file,
        "weight_path": weight_path,
        "inference_ready": True,
        "custom": True,
        "created_at": entry.get("created_at", ""),
    }


def register_custom_model(name: str, architecture: str, source_weight_path: Path) -> str:
    architecture = str(architecture).strip()
    if architecture not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"不支持的模型结构：{architecture}")

    source = Path(source_weight_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"权重文件不存在：{source}")
    if source.suffix.lower() not in {".pth", ".pt"}:
        raise ValueError("权重文件必须是 .pth 或 .pt。")

    display_name = str(name).strip() or f"{SUPPORTED_MODEL_TYPES[architecture]} 自定义模型"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    key = _unique_custom_key(architecture, display_name, timestamp)

    CUSTOM_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    destination = CUSTOM_WEIGHT_DIR / f"{key}{source.suffix.lower()}"
    shutil.copy2(source, destination)

    weight_file = destination.relative_to(LOCAL_WEIGHT_DIR).as_posix()
    entries = [entry for entry in load_custom_model_entries() if entry.get("key") != key]
    entries.append(
        {
            "key": key,
            "name": display_name,
            "architecture": architecture,
            "weight_file": weight_file,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    save_custom_model_entries(entries)
    reload_model_specs()
    return key


def remove_custom_model(model_key: str, remove_weight: bool = False) -> bool:
    entries = load_custom_model_entries()
    target_entry = next((entry for entry in entries if entry.get("key") == model_key), None)
    if target_entry is None:
        return False

    save_custom_model_entries([entry for entry in entries if entry.get("key") != model_key])
    if remove_weight:
        weight_path = _resolve_weight_file(str(target_entry.get("weight_file", "")))
        try:
            if weight_path.exists() and weight_path.is_file() and _is_subpath(weight_path, CUSTOM_WEIGHT_DIR):
                weight_path.unlink()
        except OSError:
            pass
    reload_model_specs()
    return True


def model_name(model_key: str) -> str:
    return MODEL_SPECS.get(model_key, {}).get("name", str(model_key))


def model_architecture(model_key: str) -> str:
    return MODEL_SPECS.get(model_key, {}).get("architecture", model_key)


def _resolve_weight_file(weight_file: str) -> Path:
    path = Path(weight_file)
    if path.is_absolute():
        return path

    local_path = LOCAL_WEIGHT_DIR / path
    if local_path.exists():
        return local_path
    return PACKAGED_WEIGHT_DIR / path


def _unique_custom_key(architecture: str, display_name: str, timestamp: str) -> str:
    name_part = re.sub(r"[^a-z0-9]+", "_", display_name.lower()).strip("_") or "model"
    base = f"custom_{architecture}_{name_part}_{timestamp}"
    existing = {entry.get("key") for entry in load_custom_model_entries()} | set(BASE_MODEL_SPECS)
    key = base
    index = 2
    while key in existing:
        key = f"{base}_{index}"
        index += 1
    return key


def _is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


reload_model_specs()
