import argparse
import csv
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.aide_model import AIDEModel
from models.efficientnet_model import EfficientNetB0Model
from models.model_registry import model_name
from models.resnet_model import ResNet50Model


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "Night-AIGC-Dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Verify all desktop inference models with 100 images.")
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--images_per_class", type=int, default=50)
    return parser.parse_args()


def collect_images(dataset_root: Path, images_per_class: int):
    test_root = dataset_root / "test"
    real_dir = test_root / "real"
    fake_dir = test_root / "fake"
    real_images = sorted(path for path in real_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    fake_images = sorted(path for path in fake_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)

    if len(real_images) < images_per_class or len(fake_images) < images_per_class:
        raise RuntimeError(
            f"Need {images_per_class} images per class, got real={len(real_images)}, fake={len(fake_images)}"
        )

    samples = [(path, "REAL") for path in real_images[:images_per_class]]
    samples.extend((path, "AIGC") for path in fake_images[:images_per_class])
    return samples


def build_model(model_key: str, device: str):
    if model_key == "aide":
        return AIDEModel(device=device)
    if model_key == "resnet50":
        return ResNet50Model(device=device)
    if model_key == "efficientnet_b0":
        return EfficientNetB0Model(device=device)
    raise ValueError(f"Unsupported model: {model_key}")


def evaluate_model(model_key: str, samples: list[tuple[Path, str]], device: str):
    display_name = model_name(model_key)
    print(f"\n=== {display_name} ===", flush=True)
    model = build_model(model_key, device)
    model.load()
    print(f"Loaded {display_name}: device={model.device}, load_time={model.load_time:.2f}s", flush=True)

    rows = []
    correct = 0
    total_time = 0.0
    for index, (image_path, true_label) in enumerate(samples, start=1):
        prediction = model.predict(image_path)
        pred_label = prediction["result"]
        is_correct = pred_label == true_label
        correct += int(is_correct)
        elapsed = parse_seconds(prediction["time"])
        total_time += elapsed
        rows.append(
            {
                "index": index,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "true_label": true_label,
                "prediction": pred_label,
                "correct": int(is_correct),
                "ai_probability": prediction["ai_probability"],
                "real_probability": prediction["real_probability"],
                "time": prediction["time"],
            }
        )
        print(
            f"{index:03d}/100 {image_path.name} true={true_label} pred={pred_label} "
            f"ai={prediction['ai_probability']:.6f} real={prediction['real_probability']:.6f} "
            f"time={prediction['time']}",
            flush=True,
        )

    summary = {
        "model_key": model_key,
        "model_name": display_name,
        "total": len(samples),
        "correct": correct,
        "accuracy": correct / len(samples) if samples else 0.0,
        "avg_time_seconds": total_time / len(samples) if samples else 0.0,
        "device": str(model.device),
        "load_time_seconds": model.load_time,
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, rows


def parse_seconds(value: str) -> float:
    try:
        return float(str(value).rstrip("s"))
    except ValueError:
        return 0.0


def write_outputs(output_dir: Path, run_id: str, summaries: list[dict], rows_by_model: dict[str, list[dict]]):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"model_verify_100_{run_id}_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)

    csv_paths = {}
    for model_key, rows in rows_by_model.items():
        csv_path = output_dir / f"model_verify_100_{run_id}_{model_key}.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "index",
                    "image_name",
                    "image_path",
                    "true_label",
                    "prediction",
                    "correct",
                    "ai_probability",
                    "real_probability",
                    "time",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        csv_paths[model_key] = str(csv_path)

    return summary_path, csv_paths


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    samples = collect_images(dataset_root, args.images_per_class)
    print(f"Samples: {len(samples)} ({args.images_per_class} REAL + {args.images_per_class} AIGC)")
    print(f"Dataset: {dataset_root}")
    print(f"Output: {output_dir}")

    summaries = []
    rows_by_model = {}
    for model_key in ["resnet50", "efficientnet_b0", "aide"]:
        summary, rows = evaluate_model(model_key, samples, args.device)
        summaries.append(summary)
        rows_by_model[model_key] = rows
        print(
            f"Summary {summary['model_name']}: "
            f"accuracy={summary['accuracy']:.4f}, avg_time={summary['avg_time_seconds']:.3f}s",
            flush=True,
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path, csv_paths = write_outputs(output_dir, run_id, summaries, rows_by_model)
    print(f"\nSaved summary: {summary_path}")
    for model_key, csv_path in csv_paths.items():
        print(f"Saved {model_key}: {csv_path}")


if __name__ == "__main__":
    main()
