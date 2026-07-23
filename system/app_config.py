import copy
import json
import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    APP_ROOT = Path(sys.executable).resolve().parent
    RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", APP_ROOT))
else:
    APP_ROOT = Path(__file__).resolve().parents[1]
    RESOURCE_ROOT = APP_ROOT

PROJECT_ROOT = APP_ROOT
CONFIG_PATH = APP_ROOT / "config" / "config.json"

DEFAULT_CONFIG = {
    "default_model": "aide",
    "device": "cuda",
    "paths": {
        "history": "data/history.json",
        "results": "output/results",
        "heatmaps": "output/heatmaps",
        "weights": {
            "aide": "models/weights/aide_night_best.pth",
            "resnet50": "models/weights/resnet50_ai_detector.pth",
            "efficientnet_b0": "models/weights/efficientnet_b0_ai_detector.pth",
        },
    },
    "behavior": {
        "save_history": True,
        "auto_gradcam": True,
    },
    "labels": {
        "real": 0,
        "fake": 1,
    },
}


def default_config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config() -> dict:
    source_path = CONFIG_PATH
    if not source_path.exists():
        packaged_config_path = resource_path("config/config.json")
        source_path = packaged_config_path if packaged_config_path.exists() else None

    if source_path is None:
        return default_config()

    try:
        with source_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        data = {}

    return merge_config(DEFAULT_CONFIG, data if isinstance(data, dict) else {})


def save_config(config: dict) -> dict:
    merged = merge_config(DEFAULT_CONFIG, config if isinstance(config, dict) else {})
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(merged, file, ensure_ascii=False, indent=2)
    return merged


def merge_config(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_device(device: str) -> str:
    return "cpu" if str(device).lower().startswith("cpu") else "cuda"


def resolve_path(value) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    return APP_ROOT / path


def resource_path(value) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    return RESOURCE_ROOT / path


def path_for(key: str, config: dict | None = None) -> Path:
    active_config = config or load_config()
    path_value = active_config.get("paths", {}).get(key, DEFAULT_CONFIG["paths"].get(key, ""))
    return resolve_path(path_value)


def behavior_enabled(key: str, config: dict | None = None) -> bool:
    active_config = config or load_config()
    default_value = DEFAULT_CONFIG["behavior"].get(key, False)
    return bool(active_config.get("behavior", {}).get(key, default_value))


def display_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve(strict=False).relative_to(APP_ROOT.resolve(strict=False)))
    except Exception:
        return str(path)
