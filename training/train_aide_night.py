import argparse
import csv
import datetime as dt
import json
import random
import sys
import time
import types
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AIDE_ROOT = PROJECT_ROOT / "aide_external"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "Night-AIGC-Dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "training" / "aide"
DEFAULT_PRETRAINED = PROJECT_ROOT / "models" / "weights" / "aide_night_best.pth"

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".dib", ".webp",
    ".tif", ".tiff", ".ppm", ".pgm", ".pbm", ".pnm",
}

sys.path.insert(0, str(AIDE_ROOT))

from data.dct import DCT_base_Rec_Module  # noqa: E402
from models.AIDE import AIDE  # noqa: E402

try:
    import kornia.augmentation as K
except Exception:
    K = None


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune AIDE on Night-AIGC-Dataset with local single-GPU training.")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--test_dir", default=None)
    parser.add_argument("--eval_dir", default=None)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pretrained_checkpoint", default=str(DEFAULT_PRETRAINED))
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--auto_resume", action="store_true")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--print_freq", type=int, default=50)

    parser.add_argument("--train_scope", choices=["recommended", "head", "all_except_convnext"], default="recommended")
    parser.add_argument("--class_weight", choices=["balanced", "none", "manual"], default="balanced")
    parser.add_argument("--real_weight", type=float, default=1.0)
    parser.add_argument("--fake_weight", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--disable_perturbation", action="store_true")
    parser.add_argument("--max_train_samples_per_class", type=int, default=None)
    parser.add_argument("--max_val_samples_per_class", type=int, default=None)
    parser.add_argument("--max_test_samples_per_class", type=int, default=None)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--test_after_train", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def log(message):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_split_dirs(args):
    dataset_root = Path(args.dataset_root)
    train_dir = Path(args.train_dir) if args.train_dir else dataset_root / "train"
    val_dir = Path(args.val_dir) if args.val_dir else dataset_root / "val"
    test_dir = Path(args.test_dir) if args.test_dir else dataset_root / "test"
    eval_dir = Path(args.eval_dir) if args.eval_dir else test_dir
    return train_dir, val_dir, test_dir, eval_dir


def collect_class_images(split_dir, class_name):
    direct = split_dir / class_name
    alt = split_dir / ("0_real" if class_name == "real" else "1_fake")
    class_dir = direct if direct.exists() else alt
    if not class_dir.exists():
        raise FileNotFoundError(f"Missing class directory: {class_dir}")
    return [
        path for path in sorted(class_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


class NightAIGCDataset(Dataset):
    class_to_idx = {"real": 0, "fake": 1}
    idx_to_class = {0: "real", 1: "fake"}

    def __init__(self, split_dir, is_train, image_size=256, seed=0, max_samples_per_class=None, use_perturbation=True):
        self.split_dir = Path(split_dir)
        self.is_train = is_train
        self.image_size = image_size
        self.data = []

        rng = random.Random(seed)
        for class_name, label in self.class_to_idx.items():
            images = collect_class_images(self.split_dir, class_name)
            if max_samples_per_class is not None and len(images) > max_samples_per_class:
                images = sorted(rng.sample(images, max_samples_per_class))
            self.data.extend((path, label) for path in images)

        if not self.data:
            raise FileNotFoundError(f"No images found in: {self.split_dir}")

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Compose([
            transforms.Resize([image_size, image_size], antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.dct = DCT_base_Rec_Module()
        self.perturbation = None
        if is_train and use_perturbation and K is not None:
            self.perturbation = K.container.ImageSequential(
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
                K.RandomJPEG(jpeg_quality=(30, 100), p=0.1),
            )

    def apply_perturbation(self, image):
        if self.perturbation is None:
            return image

        _, height, width = image.shape
        pad_h = (16 - height % 16) % 16
        pad_w = (16 - width % 16) % 16
        batch = image.unsqueeze(0)
        if pad_h or pad_w:
            batch = F.pad(batch, (0, pad_w, 0, pad_h), mode="reflect")

        batch = self.perturbation(batch)
        batch = batch[:, :, :height, :width]
        return batch.squeeze(0)

    def __len__(self):
        return len(self.data)

    @property
    def class_counts(self):
        real_count = sum(1 for _, label in self.data if label == 0)
        fake_count = sum(1 for _, label in self.data if label == 1)
        return {"real": real_count, "fake": fake_count}

    def __getitem__(self, index):
        image_path, label = self.data[index]
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.to_tensor(image)
            image = self.apply_perturbation(image)
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc

        views = [
            self.normalize(x_minmin),
            self.normalize(x_maxmax),
            self.normalize(x_minmin1),
            self.normalize(x_maxmax1),
            self.normalize(image),
        ]
        return torch.stack(views, dim=0), torch.tensor(label, dtype=torch.long), str(image_path)


def patch_aide_forward_for_trainable_projection(model):
    def forward(self, x):
        _, _, _, _, _ = x.shape

        x_minmin = x[:, 0]
        x_maxmax = x[:, 1]
        x_minmin1 = x[:, 2]
        x_maxmax1 = x[:, 3]
        tokens = x[:, 4]

        x_minmin = self.hpf(x_minmin)
        x_maxmax = self.hpf(x_maxmax)
        x_minmin1 = self.hpf(x_minmin1)
        x_maxmax1 = self.hpf(x_maxmax1)

        with torch.no_grad():
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=tokens.device).view(3, 1, 1)
            clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=tokens.device).view(3, 1, 1)
            dinov2_mean = torch.tensor([0.485, 0.456, 0.406], device=tokens.device).view(3, 1, 1)
            dinov2_std = torch.tensor([0.229, 0.224, 0.225], device=tokens.device).view(3, 1, 1)
            local_convnext_image_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            )
            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)

        x_0 = self.convnext_proj(local_convnext_image_feats)
        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)
        x_1 = (x_min + x_max + x_min1 + x_max1) / 4
        return self.fc(torch.cat([x_0, x_1], dim=1))

    model.forward = types.MethodType(forward, model)


def build_model(args, device):
    model = AIDE(resnet_path=None, convnext_path=None)
    patch_aide_forward_for_trainable_projection(model)
    load_model_weights(model, Path(args.pretrained_checkpoint), strict=False)
    configure_trainable_parameters(model, args.train_scope)
    model.to(device)
    return model


def strip_module_prefix(state_dict):
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_model_weights(model, checkpoint_path, strict=False):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    with checkpoint_path.open("rb") as file:
        checkpoint = torch.load(file, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(strip_module_prefix(state_dict), strict=strict)
    log(f"Loaded weights: {checkpoint_path}")
    log(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def configure_trainable_parameters(model, train_scope):
    for param in model.parameters():
        param.requires_grad = False

    train_modules = []
    if train_scope == "head":
        train_modules = ["convnext_proj", "fc"]
    elif train_scope == "recommended":
        train_modules = ["model_min", "model_max", "convnext_proj", "fc"]
    elif train_scope == "all_except_convnext":
        train_modules = ["model_min", "model_max", "convnext_proj", "fc"]

    for module_name in train_modules:
        module = getattr(model, module_name)
        for param in module.parameters():
            param.requires_grad = True

    for param in model.hpf.parameters():
        param.requires_grad = False
    for param in model.openclip_convnext_xxl.parameters():
        param.requires_grad = False


def keep_frozen_modules_in_eval(model):
    model.openclip_convnext_xxl.eval()


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator if shuffle else None,
    )


def make_class_weights(args, class_counts, device):
    if args.class_weight == "none":
        return None, {"real": 1.0, "fake": 1.0}
    if args.class_weight == "manual":
        weights = {"real": float(args.real_weight), "fake": float(args.fake_weight)}
    else:
        real_count = class_counts["real"]
        fake_count = class_counts["fake"]
        total = real_count + fake_count
        weights = {
            "real": total / (2.0 * real_count),
            "fake": total / (2.0 * fake_count),
        }
    tensor = torch.tensor([weights["real"], weights["fake"]], dtype=torch.float32, device=device)
    return tensor, weights


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def compute_metrics(y_true, y_pred, loss_sum=None, total_for_loss=None):
    rr = rf = fr = ff = 0
    for actual, pred in zip(y_true, y_pred):
        if actual == 0 and pred == 0:
            rr += 1
        elif actual == 0 and pred == 1:
            rf += 1
        elif actual == 1 and pred == 0:
            fr += 1
        elif actual == 1 and pred == 1:
            ff += 1

    total = len(y_true)
    correct = rr + ff
    real_support = rr + rf
    fake_support = fr + ff
    real_precision = safe_div(rr, rr + fr)
    real_recall = safe_div(rr, real_support)
    fake_precision = safe_div(ff, ff + rf)
    fake_recall = safe_div(ff, fake_support)
    real_f1 = safe_div(2 * real_precision * real_recall, real_precision + real_recall)
    fake_f1 = safe_div(2 * fake_precision * fake_recall, fake_precision + fake_recall)
    metrics = {
        "loss": safe_div(loss_sum, total_for_loss) if loss_sum is not None else 0.0,
        "accuracy": safe_div(correct, total),
        "balanced_accuracy": (real_recall + fake_recall) / 2.0,
        "macro_f1": (real_f1 + fake_f1) / 2.0,
        "total": total,
        "correct": correct,
        "true_real_pred_real": rr,
        "true_real_pred_fake": rf,
        "true_fake_pred_real": fr,
        "true_fake_pred_fake": ff,
        "real_precision": real_precision,
        "real_recall": real_recall,
        "real_f1": real_f1,
        "real_support": real_support,
        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "fake_f1": fake_f1,
        "fake_support": fake_support,
    }
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, args):
    model.train()
    keep_frozen_modules_in_eval(model)

    running_loss = 0.0
    total_seen = 0
    y_true = []
    y_pred = []
    optimizer.zero_grad(set_to_none=True)
    start = time.time()

    for step, (images, targets, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        loss_for_backward = loss / args.accum_steps
        if scaler is not None:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        if step % args.accum_steps == 0 or step == len(loader):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total_seen += batch_size
        preds = outputs.argmax(dim=1)
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

        if step % args.print_freq == 0 or step == len(loader):
            metrics = compute_metrics(y_true, y_pred, running_loss, total_seen)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = str(dt.timedelta(seconds=int(time.time() - start)))
            log(
                f"epoch {epoch}/{args.epochs} step {step}/{len(loader)} "
                f"loss={metrics['loss']:.6f} acc={metrics['accuracy']:.4f} lr={lr:.2e} elapsed={elapsed}"
            )

    return compute_metrics(y_true, y_pred, running_loss, total_seen)


@torch.no_grad()
def evaluate(model, loader, criterion, device, args, save_rows=False):
    model.eval()
    keep_frozen_modules_in_eval(model)

    running_loss = 0.0
    total_seen = 0
    y_true = []
    y_pred = []
    rows = []

    for step, (images, targets, paths) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total_seen += batch_size
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

        if save_rows:
            probs_cpu = probs.detach().cpu()
            preds_cpu = preds.detach().cpu().tolist()
            targets_cpu = targets.detach().cpu().tolist()
            for path, actual, pred, prob in zip(paths, targets_cpu, preds_cpu, probs_cpu):
                real_prob = float(prob[0])
                fake_prob = float(prob[1])
                rows.append({
                    "image_path": path,
                    "true_label": NightAIGCDataset.idx_to_class[int(actual)],
                    "prediction": NightAIGCDataset.idx_to_class[int(pred)],
                    "confidence": max(real_prob, fake_prob),
                    "real_probability": real_prob,
                    "fake_probability": fake_prob,
                })

        if step % max(1, args.print_freq) == 0 or step == len(loader):
            log(f"eval step {step}/{len(loader)}")

    metrics = compute_metrics(y_true, y_pred, running_loss, total_seen)
    return metrics, rows


def save_checkpoint(path, model, optimizer, scaler, epoch, best_val_acc, args, class_counts, class_weights):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "class_to_idx": NightAIGCDataset.class_to_idx,
        "class_counts": class_counts,
        "class_weights": class_weights,
        "args": vars(args),
    }
    with path.open("wb") as file:
        torch.save(payload, file)


def load_training_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    with Path(path).open("rb") as file:
        checkpoint = torch.load(file, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(strip_module_prefix(state_dict), strict=False)
    start_epoch = 1
    best_val_acc = 0.0
    if isinstance(checkpoint, dict):
        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if isinstance(checkpoint.get("epoch"), int):
            start_epoch = checkpoint["epoch"] + 1
        best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
    log(f"Resumed checkpoint: {path}")
    return start_epoch, best_val_acc


def write_metrics_csv(path, metrics):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def write_predictions_csv(path, rows):
    fieldnames = ["image_path", "true_label", "prediction", "confidence", "real_probability", "fake_probability"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_training_log(path, row):
    fieldnames = [
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_balanced_acc",
        "val_macro_f1", "real_recall", "fake_recall", "lr", "saved_best", "best_checkpoint",
        "last_checkpoint", "epoch_seconds",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_config(path, args, train_counts, val_counts, class_weights, total_params, trainable_params):
    config = {
        "args": vars(args),
        "label_mapping": NightAIGCDataset.class_to_idx,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "class_weight_mode": args.class_weight,
        "class_weights": class_weights,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "note": "recommended scope trains model_min, model_max, convnext_proj, and fc; openclip_convnext_xxl and HPF stay frozen.",
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def train(args):
    if args.dry_run:
        args.epochs = 1
        args.max_train_samples_per_class = args.max_train_samples_per_class or 2
        args.max_val_samples_per_class = args.max_val_samples_per_class or 2
        args.print_freq = 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir, val_dir, test_dir, _ = resolve_split_dirs(args)
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        log("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    log(f"Train dir: {train_dir}")
    log(f"Val dir: {val_dir}")
    log(f"Output dir: {output_dir}")
    log(f"Device: {device}")

    train_dataset = NightAIGCDataset(
        train_dir,
        is_train=True,
        image_size=args.image_size,
        seed=args.seed,
        max_samples_per_class=args.max_train_samples_per_class,
        use_perturbation=not args.disable_perturbation,
    )
    val_dataset = NightAIGCDataset(
        val_dir,
        is_train=False,
        image_size=args.image_size,
        seed=args.seed,
        max_samples_per_class=args.max_val_samples_per_class,
        use_perturbation=False,
    )
    log(f"Train counts: {train_dataset.class_counts}")
    log(f"Val counts: {val_dataset.class_counts}")

    model = build_model(args, device)
    total_params, trainable_params = count_parameters(model)
    log(f"Parameters: total={total_params}, trainable={trainable_params}")

    class_weight_tensor, class_weights = make_class_weights(args, train_dataset.class_counts, device)
    log(f"Class weights: {class_weights}")
    train_criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_path = output_dir / "aide_night_best.pth"
    last_path = output_dir / "aide_night_last.pth"
    log_path = output_dir / "aide_training_log.csv"
    save_config(output_dir / "aide_training_config.json", args, train_dataset.class_counts, val_dataset.class_counts, class_weights, total_params, trainable_params)

    start_epoch = 1
    best_val_acc = 0.0
    auto_resume_path = last_path if args.auto_resume and last_path.exists() else None
    resume_path = Path(args.resume_checkpoint) if args.resume_checkpoint else auto_resume_path
    if resume_path:
        start_epoch, best_val_acc = load_training_checkpoint(resume_path, model, optimizer, scaler, device=device)

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device.type == "cuda", args.seed)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, device.type == "cuda", args.seed)

    log(f"Start training: epochs={args.epochs}, batch_size={args.batch_size}, accum_steps={args.accum_steps}, amp={args.amp}")
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, train_criterion, optimizer, scaler, device, epoch, args)
        val_metrics, val_rows = evaluate(model, val_loader, eval_criterion, device, args, save_rows=True)

        saved_best = val_metrics["accuracy"] > best_val_acc
        if saved_best:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint(best_path, model, optimizer, scaler, epoch, best_val_acc, args, train_dataset.class_counts, class_weights)
            write_predictions_csv(output_dir / "aide_val_predictions_best.csv", val_rows)
            write_metrics_csv(output_dir / "aide_val_metrics_best.csv", val_metrics)

        save_checkpoint(last_path, model, optimizer, scaler, epoch, best_val_acc, args, train_dataset.class_counts, class_weights)
        if args.save_every_epoch:
            save_checkpoint(output_dir / f"aide_night_epoch_{epoch:03d}.pth", model, optimizer, scaler, epoch, best_val_acc, args, train_dataset.class_counts, class_weights)

        epoch_seconds = int(time.time() - epoch_start)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_balanced_acc": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "real_recall": val_metrics["real_recall"],
            "fake_recall": val_metrics["fake_recall"],
            "lr": optimizer.param_groups[0]["lr"],
            "saved_best": saved_best,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "epoch_seconds": epoch_seconds,
        }
        append_training_log(log_path, row)
        log(
            f"epoch {epoch} done train_loss={train_metrics['loss']:.6f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_balanced_acc={val_metrics['balanced_accuracy']:.4f} best={best_val_acc:.4f} "
            f"saved_best={saved_best}"
        )

    log(f"Training finished. Best checkpoint: {best_path}")
    log(f"Last checkpoint: {last_path}")
    log(f"Training log: {log_path}")

    if args.test_after_train:
        log("Running test_after_train with best checkpoint.")
        load_model_weights(model, best_path, strict=False)
        test_dataset = NightAIGCDataset(
            test_dir,
            is_train=False,
            image_size=args.image_size,
            seed=args.seed,
            max_samples_per_class=args.max_test_samples_per_class,
            use_perturbation=False,
        )
        test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device.type == "cuda", args.seed)
        test_metrics, test_rows = evaluate(model, test_loader, eval_criterion, device, args, save_rows=True)
        write_metrics_csv(output_dir / "aide_test_metrics.csv", test_metrics)
        write_predictions_csv(output_dir / "aide_test_predictions.csv", test_rows)
        log(f"Test accuracy={test_metrics['accuracy']:.6f}, balanced_accuracy={test_metrics['balanced_accuracy']:.6f}")


def evaluate_only(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, _, eval_dir = resolve_split_dirs(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    set_seed(args.seed)

    model = AIDE(resnet_path=None, convnext_path=None)
    patch_aide_forward_for_trainable_projection(model)
    checkpoint = Path(args.resume_checkpoint or args.pretrained_checkpoint)
    load_model_weights(model, checkpoint, strict=False)
    model.to(device)

    dataset = NightAIGCDataset(
        eval_dir,
        is_train=False,
        image_size=args.image_size,
        seed=args.seed,
        max_samples_per_class=args.max_test_samples_per_class,
        use_perturbation=False,
    )
    loader = make_loader(dataset, args.batch_size, False, args.num_workers, device.type == "cuda", args.seed)
    metrics, rows = evaluate(model, loader, nn.CrossEntropyLoss(), device, args, save_rows=True)
    write_metrics_csv(output_dir / "aide_eval_metrics.csv", metrics)
    write_predictions_csv(output_dir / "aide_eval_predictions.csv", rows)
    log(f"Eval accuracy={metrics['accuracy']:.6f}, balanced_accuracy={metrics['balanced_accuracy']:.6f}")


def main():
    args = parse_args()
    if args.accum_steps < 1:
        raise ValueError("--accum_steps must be >= 1")
    if args.mode == "train":
        train(args)
    else:
        evaluate_only(args)


if __name__ == "__main__":
    main()
