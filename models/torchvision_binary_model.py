from pathlib import Path
import threading
import time

import torch
from PIL import Image, ImageFile
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet50


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".dib", ".webp", ".tif", ".tiff"}


def build_resnet50():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def build_efficientnet_b0():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


class TorchvisionBinaryModel:
    def __init__(
        self,
        model_key: str,
        model_name: str,
        weight_path: Path,
        device="cuda",
        architecture: str | None = None,
    ):
        self.model_key = model_key
        self.architecture = architecture or model_key
        self.model_name = model_name
        self.weight_path = Path(weight_path)
        self.requested_device = device
        self.device = torch.device("cpu")
        self.model = None
        self.class_to_idx = {"real": 0, "fake": 1}
        self.load_time = None
        self._predict_lock = threading.Lock()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self):
        if not self.weight_path.exists():
            raise FileNotFoundError(f"Weight file does not exist: {self.weight_path}")

        start_time = time.perf_counter()
        self.device = self._resolve_device()
        model = self._build_model()
        checkpoint = torch.load(self.weight_path, map_location="cpu")
        state_dict = checkpoint.get("model_state", checkpoint.get("model", checkpoint))
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("class_to_idx"), dict):
            self.class_to_idx = dict(checkpoint["class_to_idx"])

        model.load_state_dict(strip_module_prefix(state_dict))
        model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        self.model = model
        self.load_time = time.perf_counter() - start_time
        return self

    def predict(self, image_path):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {image_path}")

        with self._predict_lock:
            if not self.is_loaded:
                self.load()

            start_time = time.perf_counter()
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logits = self.model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()

            real_index = self.class_to_idx.get("real", 0)
            ai_index = self.class_to_idx.get("fake", 1)
            real_probability = float(probabilities[real_index])
            ai_probability = float(probabilities[ai_index])

            return {
                "result": "AIGC" if ai_probability >= real_probability else "REAL",
                "ai_probability": round(ai_probability, 6),
                "real_probability": round(real_probability, 6),
                "time": f"{time.perf_counter() - start_time:.3f}s",
            }

    def _build_model(self):
        if self.architecture == "resnet50":
            return build_resnet50()
        if self.architecture == "efficientnet_b0":
            return build_efficientnet_b0()
        raise ValueError(f"Unsupported model architecture: {self.architecture}")

    def _resolve_device(self):
        if self.requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(self.requested_device).startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.requested_device)


def strip_module_prefix(state_dict):
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
