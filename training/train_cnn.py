import argparse
import csv
import random
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DIR = PROJECT_ROOT / "datasets" / "Night-AIGC-Dataset" / "test"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "output" / "training" / "cnn"
IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".dib", ".webp",
    ".tif", ".tiff", ".ppm", ".pgm", ".pbm", ".pnm",
}
CLASS_ALIASES = {
    "0_real": "real",
    "real": "real",
    "1_fake": "fake",
    "fake": "fake",
}
CANONICAL_CLASS_TO_IDX = {"real": 0, "fake": 1}
CANONICAL_CLASSES = ["real", "fake"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train or run ResNet50/EfficientNet-B0 for AI image detection.")
    parser.add_argument("--mode", choices=["train", "predict", "evaluate"], required=True, help="train、predict 或 evaluate。")
    parser.add_argument("--model", choices=["resnet50", "efficientnet_b0", "ensemble"], default="resnet50")
    parser.add_argument("--train_dir", default="", help="训练集目录，需包含 real/fake 或 0_real/1_fake 两个子文件夹。")
    parser.add_argument("--val_dir", default="", help="验证集目录，需包含 real/fake 或 0_real/1_fake；不填则用训练集做验证。")
    parser.add_argument("--test_dir", default="", help="测试集目录，需包含 real/fake 或 0_real/1_fake 两个子文件夹。")
    parser.add_argument("--image_path", default=str(DEFAULT_TEST_DIR), help="预测图片文件或图片文件夹。")
    parser.add_argument("--result_dir", default=str(DEFAULT_RESULT_DIR), help="输出目录。")
    parser.add_argument("--checkpoint", default="", help="单模型预测时指定权重；不填则使用默认路径。")
    parser.add_argument("--resnet_checkpoint", default="", help="ensemble 模式下 ResNet50 权重路径。")
    parser.add_argument("--efficientnet_checkpoint", default="", help="ensemble 模式下 EfficientNet-B0 权重路径。")
    parser.add_argument("--pretrained", action="store_true", help="训练时使用 ImageNet 预训练权重；首次使用可能需要联网下载。")
    parser.add_argument("--test_after_train", action="store_true", help="训练结束后加载验证集最优权重并在 --test_dir 上评估。")
    parser.add_argument("--freeze_backbone", action="store_true", help="只训练最后分类层，用于快速微调。")
    parser.add_argument("--max_images_per_class", type=int, default=0, help="每个类别最多使用多少张图；0 表示全量训练。")
    parser.add_argument("--class_weight", choices=["none", "balanced", "manual"], default="none", help="训练 loss 类别权重。balanced 按训练集数量自动计算。")
    parser.add_argument("--real_weight", type=float, default=1.0, help="--class_weight manual 时 real 类 loss 权重。")
    parser.add_argument("--fake_weight", type=float, default=1.0, help="--class_weight manual 时 fake 类 loss 权重。")
    parser.add_argument("--auto_resume", action="store_true", help="训练时自动从 result_dir 中的 last checkpoint 续训。")
    parser.add_argument("--resume_checkpoint", default="", help="训练时指定断点 checkpoint 继续训练。")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def get_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动改用 CPU。")
        name = "cpu"
    return torch.device(name)


def get_transform(train):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(model_name, pretrained=False):
    if model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=None)
        if weights is not None:
            # torchvision 0.16 may expect the wrong hash prefix for this file.
            state_dict = load_state_dict_from_url(weights.url, progress=True, check_hash=False)
            model.load_state_dict(state_dict)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def freeze_backbone(model, model_name):
    for param in model.parameters():
        param.requires_grad = False

    if model_name == "resnet50":
        for param in model.fc.parameters():
            param.requires_grad = True
        return

    if model_name == "efficientnet_b0":
        for param in model.classifier.parameters():
            param.requires_grad = True
        return

    raise ValueError(f"Unsupported model: {model_name}")


def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def default_checkpoint_path(result_dir, model_name):
    return Path(result_dir) / f"{model_name}_ai_detector.pth"


def last_checkpoint_path(result_dir, model_name):
    return Path(result_dir) / f"{model_name}_last_checkpoint.pth"


def training_log_path(result_dir, model_name):
    return Path(result_dir) / f"{model_name}_training_log.csv"


def init_training_log(path, append=False):
    if append and path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "saved_best",
            "best_val_acc",
            "best_checkpoint_path",
            "last_checkpoint_path",
        ])


def append_training_log(
    path,
    epoch,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    saved_best,
    best_val_acc,
    checkpoint_path,
    last_checkpoint,
):
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
            int(saved_best),
            f"{best_val_acc:.6f}",
            str(checkpoint_path),
            str(last_checkpoint),
        ])


def save_checkpoint(
    path,
    model_name,
    model,
    class_to_idx,
    epoch,
    val_acc,
    optimizer=None,
    best_val_acc=None,
    train_config=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": model_name,
        "model_state": model.state_dict(),
        "class_to_idx": class_to_idx,
        "epoch": epoch,
        "val_acc": val_acc,
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if best_val_acc is not None:
        checkpoint["best_val_acc"] = best_val_acc
    if train_config is not None:
        checkpoint["train_config"] = train_config
    with path.open("wb") as file:
        torch.save(checkpoint, file)


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def resolve_resume_checkpoint(args, result_dir):
    if args.resume_checkpoint:
        return Path(args.resume_checkpoint)
    if args.auto_resume:
        path = last_checkpoint_path(result_dir, args.model)
        if path.exists():
            return path
        print(f"No last checkpoint found for auto resume: {path}")
    return None


def load_training_checkpoint(path, model_name, model, optimizer, device):
    checkpoint = torch.load(path, map_location="cpu")
    saved_model_name = checkpoint.get("model_name", model_name)
    if saved_model_name != model_name:
        raise ValueError(f"断点属于 {saved_model_name}，但当前选择的是 {model_name}。")

    model.load_state_dict(checkpoint["model_state"])
    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        move_optimizer_state_to_device(optimizer, device)
    else:
        print("Resume checkpoint does not contain optimizer state; optimizer will start fresh.")

    epoch = int(checkpoint.get("epoch", 0))
    best_val_acc = float(checkpoint.get("best_val_acc", checkpoint.get("val_acc", -1.0)))
    return epoch + 1, best_val_acc


def load_checkpoint(path, model_name, device):
    checkpoint = torch.load(path, map_location="cpu")
    saved_model_name = checkpoint.get("model_name", model_name)
    if saved_model_name != model_name:
        raise ValueError(f"权重属于 {saved_model_name}，但当前选择的是 {model_name}。")
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def image_file_count(path):
    return sum(
        1 for item in Path(path).rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_dataset_root(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {root}")
    if not root.is_dir():
        raise ValueError(f"数据集路径必须是目录: {root}")

    class_dirs = {item.name: item for item in root.iterdir() if item.is_dir()}
    canonical_dirs = {}
    unsupported = []
    for class_name, class_dir in class_dirs.items():
        canonical_name = CLASS_ALIASES.get(class_name)
        if canonical_name is None:
            unsupported.append(class_name)
            continue
        if canonical_name in canonical_dirs:
            raise ValueError(
                f"{root} 中同时存在多个表示 {canonical_name} 的目录，"
                f"请只保留一种命名方式: real/fake 或 0_real/1_fake。"
            )
        canonical_dirs[canonical_name] = class_dir

    missing = [name for name in CANONICAL_CLASSES if name not in canonical_dirs]
    if missing:
        raise ValueError(
            f"{root} 缺少类别目录: {missing}。需要 real/fake 或 0_real/1_fake。"
        )

    empty = [
        f"{class_name}: {class_dir}"
        for class_name, class_dir in canonical_dirs.items()
        if image_file_count(class_dir) == 0
    ]
    if empty:
        raise ValueError(
            "以下类别没有图片，不能训练真实/AI生成二分类模型: "
            + "; ".join(empty)
        )

    if unsupported:
        print(f"忽略不支持的子目录: {sorted(unsupported)}")


def load_dataset(root, train):
    validate_dataset_root(root)
    dataset = datasets.ImageFolder(
        root,
        transform=get_transform(train=train),
        is_valid_file=lambda path: Path(path).suffix.lower() in IMAGE_SUFFIXES,
    )
    normalize_dataset_labels(dataset)
    return dataset


def normalize_dataset_labels(dataset):
    original_class_to_idx = dataset.class_to_idx.copy()
    remap = {}
    for class_name, original_idx in original_class_to_idx.items():
        canonical_name = CLASS_ALIASES.get(class_name)
        if canonical_name is None:
            raise ValueError(f"不支持的类别目录: {class_name}")
        remap[original_idx] = CANONICAL_CLASS_TO_IDX[canonical_name]

    dataset.samples = [(path, remap[label]) for path, label in dataset.samples]
    dataset.imgs = dataset.samples
    dataset.targets = [label for _, label in dataset.samples]
    dataset.classes = CANONICAL_CLASSES.copy()
    dataset.class_to_idx = CANONICAL_CLASS_TO_IDX.copy()


def limit_dataset_per_class(dataset, max_images_per_class, seed):
    if max_images_per_class <= 0:
        return

    rng = random.Random(seed)
    by_class = {idx: [] for idx in CANONICAL_CLASS_TO_IDX.values()}
    for sample in dataset.samples:
        by_class[sample[1]].append(sample)

    limited_samples = []
    for class_idx in sorted(by_class):
        samples = by_class[class_idx]
        if len(samples) > max_images_per_class:
            samples = rng.sample(samples, max_images_per_class)
        limited_samples.extend(samples)

    dataset.samples = limited_samples
    dataset.imgs = dataset.samples
    dataset.targets = [label for _, label in dataset.samples]


def dataset_counts(dataset):
    counts = {class_name: 0 for class_name in CANONICAL_CLASSES}
    idx_to_class = {idx: name for name, idx in CANONICAL_CLASS_TO_IDX.items()}
    for label in dataset.targets:
        counts[idx_to_class[label]] += 1
    return counts


def build_class_weight(args, train_dataset, device):
    if args.class_weight == "none":
        return None, None

    counts = dataset_counts(train_dataset)
    if any(counts[class_name] <= 0 for class_name in CANONICAL_CLASSES):
        raise ValueError(f"类别数量必须大于 0 才能使用 loss 权重，当前: {counts}")

    if args.class_weight == "balanced":
        total = sum(counts.values())
        weight_by_class = {
            class_name: total / (len(CANONICAL_CLASSES) * counts[class_name])
            for class_name in CANONICAL_CLASSES
        }
    else:
        if args.real_weight <= 0 or args.fake_weight <= 0:
            raise ValueError("--real_weight 和 --fake_weight 必须大于 0。")
        weight_by_class = {
            "real": args.real_weight,
            "fake": args.fake_weight,
        }

    weights = torch.tensor(
        [weight_by_class[class_name] for class_name in CANONICAL_CLASSES],
        dtype=torch.float32,
        device=device,
    )
    return weights, weight_by_class


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += labels.numel()
            loss_sum += float(loss.item()) * labels.numel()

    return loss_sum / max(total, 1), correct / max(total, 1)


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def load_models_for_evaluation(args, result_dir, device):
    models = []
    if args.model == "ensemble":
        resnet_ckpt = Path(args.resnet_checkpoint) if args.resnet_checkpoint else default_checkpoint_path(result_dir, "resnet50")
        efficientnet_ckpt = (
            Path(args.efficientnet_checkpoint)
            if args.efficientnet_checkpoint
            else default_checkpoint_path(result_dir, "efficientnet_b0")
        )
        models.append(("resnet50", load_checkpoint(resnet_ckpt, "resnet50", device)))
        models.append(("efficientnet_b0", load_checkpoint(efficientnet_ckpt, "efficientnet_b0", device)))
        return models

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(result_dir, args.model)
    models.append((args.model, load_checkpoint(checkpoint_path, args.model, device)))
    return models


def evaluate_models(models, dataset, loader, device):
    for _, model in models:
        model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0
    confusion = [[0, 0], [0, 0]]
    prediction_rows = []
    idx_to_class = {idx: name for name, idx in CANONICAL_CLASS_TO_IDX.items()}
    sample_offset = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            probs_list = []
            for _, model in models:
                logits = model(images)
                probs_list.append(torch.softmax(logits, dim=1))
            probs = torch.stack(probs_list, dim=0).mean(dim=0)
            preds = probs.argmax(dim=1)

            batch_size = labels.numel()
            batch_loss = -torch.log(
                probs.gather(1, labels.unsqueeze(1)).clamp_min(1e-12)
            ).sum()
            loss_sum += float(batch_loss.item())
            correct += int((preds == labels).sum().item())
            total += batch_size

            labels_cpu = labels.detach().cpu().tolist()
            preds_cpu = preds.detach().cpu().tolist()
            probs_cpu = probs.detach().cpu().tolist()

            for index in range(batch_size):
                true_idx = int(labels_cpu[index])
                pred_idx = int(preds_cpu[index])
                confusion[true_idx][pred_idx] += 1
                image_path = dataset.samples[sample_offset + index][0]
                real_prob = float(probs_cpu[index][0])
                fake_prob = float(probs_cpu[index][1])
                prediction_rows.append(
                    [
                        image_path,
                        idx_to_class[true_idx],
                        idx_to_class[pred_idx],
                        f"{max(real_prob, fake_prob):.6f}",
                        f"{real_prob:.6f}",
                        f"{fake_prob:.6f}",
                    ]
                )
            sample_offset += batch_size

    metrics = build_metrics(loss_sum / max(total, 1), correct, total, confusion)
    return metrics, prediction_rows


def build_metrics(loss, correct, total, confusion):
    metrics = {
        "loss": loss,
        "accuracy": safe_divide(correct, total),
        "total": total,
        "correct": correct,
        "confusion_true_real_pred_real": confusion[0][0],
        "confusion_true_real_pred_fake": confusion[0][1],
        "confusion_true_fake_pred_real": confusion[1][0],
        "confusion_true_fake_pred_fake": confusion[1][1],
    }

    recalls = []
    f1_scores = []
    for class_name, class_idx in CANONICAL_CLASS_TO_IDX.items():
        tp = confusion[class_idx][class_idx]
        fp = sum(confusion[row][class_idx] for row in range(2) if row != class_idx)
        fn = sum(confusion[class_idx][col] for col in range(2) if col != class_idx)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        metrics[f"{class_name}_precision"] = precision
        metrics[f"{class_name}_recall"] = recall
        metrics[f"{class_name}_f1"] = f1
        metrics[f"{class_name}_support"] = sum(confusion[class_idx])
        recalls.append(recall)
        f1_scores.append(f1)

    metrics["balanced_accuracy"] = sum(recalls) / len(recalls)
    metrics["macro_f1"] = sum(f1_scores) / len(f1_scores)
    return metrics


def write_evaluation_outputs(result_dir, model_name, eval_name, metrics, prediction_rows):
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / f"{model_name}_{eval_name}_metrics.csv"
    predictions_path = result_dir / f"{model_name}_{eval_name}_predictions.csv"

    with metrics_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    with predictions_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_path",
            "true_label",
            "prediction",
            "confidence",
            "real_probability",
            "fake_probability",
        ])
        writer.writerows(prediction_rows)

    return metrics_path, predictions_path


def print_evaluation_summary(eval_name, metrics, metrics_path, predictions_path):
    print(f"{eval_name} metrics:")
    print(f"  loss={metrics['loss']:.4f}")
    print(f"  accuracy={metrics['accuracy']:.4f}")
    print(f"  balanced_accuracy={metrics['balanced_accuracy']:.4f}")
    print(f"  macro_f1={metrics['macro_f1']:.4f}")
    print(
        "  confusion_matrix true->pred: "
        f"real->real={metrics['confusion_true_real_pred_real']}, "
        f"real->fake={metrics['confusion_true_real_pred_fake']}, "
        f"fake->real={metrics['confusion_true_fake_pred_real']}, "
        f"fake->fake={metrics['confusion_true_fake_pred_fake']}"
    )
    print(
        "  real: "
        f"precision={metrics['real_precision']:.4f}, "
        f"recall={metrics['real_recall']:.4f}, "
        f"f1={metrics['real_f1']:.4f}, "
        f"support={metrics['real_support']}"
    )
    print(
        "  fake: "
        f"precision={metrics['fake_precision']:.4f}, "
        f"recall={metrics['fake_recall']:.4f}, "
        f"f1={metrics['fake_f1']:.4f}, "
        f"support={metrics['fake_support']}"
    )
    print(f"  metrics_csv={metrics_path}")
    print(f"  predictions_csv={predictions_path}")


def evaluate_saved_models(args, eval_dir, eval_name="test"):
    device = get_device(args.device)
    result_dir = Path(args.result_dir)
    dataset = load_dataset(eval_dir, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    models = load_models_for_evaluation(args, result_dir, device)
    metrics, prediction_rows = evaluate_models(models, dataset, loader, device)
    metrics_path, predictions_path = write_evaluation_outputs(
        result_dir,
        args.model,
        eval_name,
        metrics,
        prediction_rows,
    )

    print(f"Evaluate model: {args.model}")
    print(f"Evaluate dir: {eval_dir}")
    print(f"Evaluate images: {len(dataset)}")
    print(f"Evaluate counts: {dataset_counts(dataset)}")
    print_evaluation_summary(eval_name, metrics, metrics_path, predictions_path)


def train(args):
    if args.model == "ensemble":
        raise ValueError("ensemble 只用于 predict。训练时请选择 resnet50 或 efficientnet_b0。")
    if not args.train_dir:
        raise ValueError("训练模式必须提供 --train_dir。")

    device = get_device(args.device)
    result_dir = Path(args.result_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(result_dir, args.model)
    last_path = last_checkpoint_path(result_dir, args.model)
    log_path = training_log_path(result_dir, args.model)
    resume_path = resolve_resume_checkpoint(args, result_dir)

    train_dataset = load_dataset(args.train_dir, train=True)
    val_dataset = load_dataset(args.val_dir or args.train_dir, train=False)
    limit_dataset_per_class(train_dataset, args.max_images_per_class, args.seed)
    limit_dataset_per_class(val_dataset, args.max_images_per_class, args.seed + 1)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )

    model = build_model(args.model, pretrained=args.pretrained)
    if args.freeze_backbone:
        freeze_backbone(model, args.model)
    model = model.to(device)
    class_weight_tensor, class_weight_info = build_class_weight(args, train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
    )

    start_epoch = 1
    best_acc = -1.0
    if resume_path:
        if not resume_path.exists():
            raise FileNotFoundError(f"断点 checkpoint 不存在: {resume_path}")
        start_epoch, best_acc = load_training_checkpoint(resume_path, args.model, model, optimizer, device)
        print(f"Resume from checkpoint: {resume_path}")
        print(f"Resume start epoch: {start_epoch}")
        print(f"Resume best val_acc: {best_acc:.4f}")

    init_training_log(log_path, append=resume_path is not None)
    train_config = {
        "class_weight": args.class_weight,
        "class_weights": class_weight_info or {"real": 1.0, "fake": 1.0},
        "freeze_backbone": args.freeze_backbone,
        "max_images_per_class": args.max_images_per_class,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    print(f"Train counts: {dataset_counts(train_dataset)}")
    print(f"Val counts: {dataset_counts(val_dataset)}")
    print(f"Class mapping: {train_dataset.class_to_idx}")
    print(f"Class weight mode: {args.class_weight}")
    print(f"Class weights: {class_weight_info or {'real': 1.0, 'fake': 1.0}}")
    print(f"Trainable parameters: {count_trainable_parameters(model)}")
    print(f"Save checkpoint: {checkpoint_path}")
    print(f"Save last checkpoint: {last_path}")
    print(f"Training log: {log_path}")

    if start_epoch > args.epochs:
        print(f"Start epoch {start_epoch} is greater than --epochs {args.epochs}; no training epochs will run.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.numel()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        saved_best = False
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                checkpoint_path,
                args.model,
                model,
                train_dataset.class_to_idx,
                epoch,
                val_acc,
                optimizer=optimizer,
                best_val_acc=best_acc,
                train_config=train_config,
            )
            saved_best = True
            print(f"Saved best checkpoint, val_acc={val_acc:.4f}")

        save_checkpoint(
            last_path,
            args.model,
            model,
            train_dataset.class_to_idx,
            epoch,
            val_acc,
            optimizer=optimizer,
            best_val_acc=best_acc,
            train_config=train_config,
        )
        print(f"Saved last checkpoint: {last_path}")

        append_training_log(
            log_path,
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            saved_best,
            best_acc,
            checkpoint_path,
            last_path,
        )

    if args.test_after_train:
        if not args.test_dir:
            raise ValueError("使用 --test_after_train 时必须提供 --test_dir。")
        print("Training finished. Loading best checkpoint for test evaluation.")
        evaluate_saved_models(args, args.test_dir, eval_name="test")


def find_images(path):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {path}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    images = [
        item for item in sorted(path.rglob("*"))
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise FileNotFoundError(f"No images found in: {path}")
    return images


def predict_one(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0].detach().cpu()
    return probs


def predict(args):
    device = get_device(args.device)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    image_paths = find_images(args.image_path)
    transform = get_transform(train=False)

    models = []
    if args.model == "ensemble":
        resnet_ckpt = Path(args.resnet_checkpoint) if args.resnet_checkpoint else default_checkpoint_path(result_dir, "resnet50")
        efficientnet_ckpt = (
            Path(args.efficientnet_checkpoint)
            if args.efficientnet_checkpoint
            else default_checkpoint_path(result_dir, "efficientnet_b0")
        )
        models.append(("resnet50", load_checkpoint(resnet_ckpt, "resnet50", device)))
        models.append(("efficientnet_b0", load_checkpoint(efficientnet_ckpt, "efficientnet_b0", device)))
        csv_path = result_dir / "ResNet50_EfficientNetB0_ensemble判断结果.csv"
    else:
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(result_dir, args.model)
        models.append((args.model, load_checkpoint(checkpoint_path, args.model, device)))
        csv_path = result_dir / f"{args.model}_判断结果.csv"

    print(f"Images: {len(image_paths)}")
    print(f"Device: {device}")
    print(f"Result: {csv_path}")

    rows = [["image_path", "model", "prediction", "confidence", "real_probability", "fake_probability"]]
    for image_path in image_paths:
        probs_list = [predict_one(model, image_path, transform, device) for _, model in models]
        probs = torch.stack(probs_list, dim=0).mean(dim=0)
        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        confidence = max(real_prob, fake_prob)
        label = "AI生成" if fake_prob >= 0.5 else "真实图片"
        rows.append([
            str(image_path),
            args.model,
            label,
            f"{confidence:.6f}",
            f"{real_prob:.6f}",
            f"{fake_prob:.6f}",
        ])
        print(
            f"{image_path.name} -> {label} | "
            f"confidence={confidence:.6f}, real={real_prob:.6f}, fake={fake_prob:.6f}"
        )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        csv.writer(file).writerows(rows)


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    else:
        if not args.test_dir:
            raise ValueError("evaluate 模式必须提供 --test_dir。")
        evaluate_saved_models(args, args.test_dir, eval_name="test")


if __name__ == "__main__":
    main()
