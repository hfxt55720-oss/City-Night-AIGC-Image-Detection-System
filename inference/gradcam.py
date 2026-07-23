from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEATMAP_DIR = PROJECT_ROOT / "output" / "heatmaps"
CLASS_NAMES = ["REAL", "AIGC"]


class GradCAM:
    def __init__(self, model, target_layer, call_index=0):
        self.model = model
        self.target_layer = target_layer
        self.call_index = call_index
        self.records = []
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        entry = {"activation": output}

        def save_gradient(gradient):
            entry["gradient"] = gradient

        output.register_hook(save_gradient)
        self.records.append(entry)

    def remove(self):
        self.handle.remove()

    def __call__(self, tensor, class_index):
        self.records = []
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        score = logits[:, class_index].sum()
        score.backward()

        if not self.records:
            raise RuntimeError("No activation captured. Check target layer.")
        if self.call_index >= len(self.records):
            raise RuntimeError(f"Target layer was called {len(self.records)} times, but call_index={self.call_index}.")

        record = self.records[self.call_index]
        activation = record["activation"].detach()
        gradient = record["gradient"].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, logits.detach()


def generate_gradcam(model_wrapper, image_path, output_dir=DEFAULT_HEATMAP_DIR, alpha=0.45, target_class="predicted"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)

    model = model_wrapper.model
    if model is None:
        raise RuntimeError("Model is not loaded.")

    model_key = getattr(model_wrapper, "model_key", None)
    model_architecture = getattr(model_wrapper, "architecture", None) or model_key
    model_name = getattr(model_wrapper, "model_name", None) or _display_model_name(model_key)
    device = model_wrapper.device
    target_layer, target_layer_name, call_index = choose_target_layer(model, model_architecture)
    tensor, display = build_model_input(model_wrapper, image_path, model_architecture)
    target_index, probabilities = choose_target_index(model, tensor, target_class)

    lock = getattr(model_wrapper, "_predict_lock", None)
    with _maybe_lock(lock), temporarily_enable_grad(model):
        model.eval()
        cam_runner = GradCAM(model, target_layer, call_index=call_index)
        try:
            with torch.enable_grad():
                cam, logits = cam_runner(tensor.to(device), target_index)
                probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
        finally:
            cam_runner.remove()

    real_probability = float(probabilities[0])
    ai_probability = float(probabilities[1])
    predicted_index = int(probabilities.argmax().item())

    heatmap = colorize_heatmap(cam, display.size)
    overlay = make_overlay(display, heatmap, alpha)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = f"{safe_stem(image_path)}_{model_key or 'model'}_{CLASS_NAMES[target_index].lower()}_{timestamp}"
    original_path = output_dir / f"{prefix}_original.jpg"
    heatmap_path = output_dir / f"{prefix}_heatmap.jpg"
    overlay_path = output_dir / f"{prefix}_overlay.jpg"

    display.convert("RGB").save(original_path, quality=95)
    heatmap.convert("RGB").save(heatmap_path, quality=95)
    overlay.convert("RGB").save(overlay_path, quality=95)

    return {
        "model_name": model_name,
        "target_layer": target_layer_name,
        "target_class": CLASS_NAMES[target_index],
        "prediction": CLASS_NAMES[predicted_index],
        "ai_probability": round(ai_probability, 6),
        "real_probability": round(real_probability, 6),
        "original_path": str(original_path),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
    }


def choose_target_layer(model, model_key):
    if model_key == "resnet50":
        return model.layer4[-1], "resnet50_layer4_last", 0
    if model_key == "efficientnet_b0":
        return model.features[-1], "efficientnet_b0_features_last", 0
    if model_key == "aide":
        return model.model_min.layer4[-1], "aide_model_min_layer4_last", 0
    raise ValueError(f"Unsupported Grad-CAM model: {model_key}")


def build_model_input(model_wrapper, image_path: Path, model_key: str):
    image = Image.open(image_path).convert("RGB")

    if model_key in {"resnet50", "efficientnet_b0"}:
        display = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
        ])(image)
        tensor = model_wrapper.transform(image).unsqueeze(0).to(model_wrapper.device)
        return tensor, display

    if model_key == "aide":
        if model_wrapper.dct_module is None:
            model_wrapper.load()
        image_tensor = model_wrapper.to_tensor(image)
        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = model_wrapper.dct_module(image_tensor)
        views = [
            model_wrapper.normalize(x_minmin),
            model_wrapper.normalize(x_maxmax),
            model_wrapper.normalize(x_minmin1),
            model_wrapper.normalize(x_maxmax1),
            model_wrapper.normalize(image_tensor),
        ]
        tensor = torch.stack(views, dim=0).unsqueeze(0).to(model_wrapper.device)
        display = image.resize((model_wrapper.image_size, model_wrapper.image_size), Image.Resampling.BILINEAR)
        return tensor, display

    raise ValueError(f"Unsupported Grad-CAM model: {model_key}")


def choose_target_index(model, tensor, target_class):
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()

    if target_class == "predicted":
        return int(probabilities.argmax().item()), probabilities
    if target_class == "REAL":
        return 0, probabilities
    if target_class == "AIGC":
        return 1, probabilities
    raise ValueError(f"Unsupported target class: {target_class}")


def colorize_heatmap(cam, size):
    import matplotlib

    heatmap = Image.fromarray(np.uint8(cam * 255)).resize(size, Image.Resampling.BILINEAR)
    heatmap_np = np.asarray(heatmap).astype(np.float32) / 255.0
    if hasattr(matplotlib, "colormaps"):
        cmap = matplotlib.colormaps.get_cmap("jet")
    else:
        import matplotlib.cm as cm

        cmap = cm.get_cmap("jet")
    colored = cmap(heatmap_np)[:, :, :3]
    return Image.fromarray(np.uint8(colored * 255))


def make_overlay(display_image, heatmap, alpha):
    base = display_image.convert("RGB")
    base_np = np.asarray(base).astype(np.float32)
    heat_np = np.asarray(heatmap).astype(np.float32)
    overlay = np.clip((1.0 - alpha) * base_np + alpha * heat_np, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


@contextmanager
def temporarily_enable_grad(model):
    states = [parameter.requires_grad for parameter in model.parameters()]
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        yield
    finally:
        for parameter, state in zip(model.parameters(), states):
            parameter.requires_grad_(state)


@contextmanager
def _maybe_lock(lock):
    if lock is None:
        yield
        return
    with lock:
        yield


def _display_model_name(model_key):
    if model_key == "aide":
        return "AIDE-Night"
    if model_key == "resnet50":
        return "ResNet50"
    if model_key == "efficientnet_b0":
        return "EfficientNet-B0"
    return str(model_key)


def safe_stem(path):
    stem = Path(path).stem
    keep = []
    for char in stem:
        keep.append(char if char.isalnum() or char in "-_" else "_")
    return "".join(keep)[:80] or "image"
