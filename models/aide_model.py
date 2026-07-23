from pathlib import Path
import importlib.util
import sys
import threading
import time
import types

import torch
from PIL import Image, ImageFile
from torchvision import transforms

from system.app_config import resource_path


ImageFile.LOAD_TRUNCATED_IMAGES = True

PACKAGED_AIDE_ROOT = resource_path("aide_external")
PACKAGED_WEIGHT = resource_path("models/weights/aide_night_best.pth")
DEFAULT_AIDE_ROOT = PACKAGED_AIDE_ROOT
DEFAULT_WEIGHT = PACKAGED_WEIGHT

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".jfif",
    ".png",
    ".bmp",
    ".dib",
    ".webp",
    ".tif",
    ".tiff",
    ".ppm",
    ".pgm",
    ".pbm",
    ".pnm",
}


def _ensure_namespace_package(package_name: str, package_dir: Path):
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_dir)]
        sys.modules[package_name] = package
    return package


def _load_module(module_name: str, file_path: Path):
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_aide_components(aide_root: Path):
    models_dir = aide_root / "models"
    data_dir = aide_root / "data"
    aide_file = models_dir / "AIDE.py"
    dct_file = data_dir / "dct.py"

    if not aide_file.exists():
        raise FileNotFoundError(f"AIDE model file does not exist: {aide_file}")
    if not dct_file.exists():
        raise FileNotFoundError(f"AIDE DCT file does not exist: {dct_file}")

    _ensure_namespace_package("_aide_external_models", models_dir)
    _ensure_namespace_package("_aide_external_data", data_dir)

    aide_module = _load_module("_aide_external_models.AIDE", aide_file)
    dct_module = _load_module("_aide_external_data.dct", dct_file)
    return aide_module.AIDE, dct_module.DCT_base_Rec_Module


def _strip_module_prefix(state_dict):
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _patch_aide_forward_for_inference(model):
    def forward(self, x):
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
            convnext_features = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            )
            convnext_features = self.avgpool(convnext_features).view(tokens.size(0), -1)

        x_0 = self.convnext_proj(convnext_features)
        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)
        x_1 = (x_min + x_max + x_min1 + x_max1) / 4
        return self.fc(torch.cat([x_0, x_1], dim=1))

    model.forward = types.MethodType(forward, model)


class AIDEModel:
    def __init__(
        self,
        weight_path=None,
        device="cuda",
        aide_root=None,
        image_size=256,
        model_key="aide",
        model_name="AIDE-Night",
    ):
        self.model_key = model_key
        self.architecture = "aide"
        self.model_name = model_name
        self.weight_path = Path(weight_path) if weight_path is not None else PACKAGED_WEIGHT
        self.aide_root = Path(aide_root) if aide_root is not None else self._default_aide_root()
        self.requested_device = device
        self.device = torch.device("cpu")
        self.image_size = image_size
        self.model = None
        self.dct_module = None
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Compose(
            [
                transforms.Resize([image_size, image_size], antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.class_to_idx = {"real": 0, "fake": 1}
        self.missing_keys = []
        self.unexpected_keys = []
        self.load_time = None
        self._predict_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.dct_module is not None

    def _resolve_weight_path(self) -> Path:
        if self.weight_path.exists():
            return self.weight_path
        if PACKAGED_WEIGHT.exists():
            return PACKAGED_WEIGHT
        raise FileNotFoundError(f"AIDE weight file does not exist: {self.weight_path}")

    def _default_aide_root(self) -> Path:
        if PACKAGED_AIDE_ROOT.exists():
            return PACKAGED_AIDE_ROOT
        return DEFAULT_AIDE_ROOT

    def _resolve_device(self):
        if self.requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(self.requested_device).startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.requested_device)

    def load(self):
        start_time = time.perf_counter()
        weight_path = self._resolve_weight_path()
        if not self.aide_root.exists():
            raise FileNotFoundError(f"AIDE root does not exist: {self.aide_root}")

        self.device = self._resolve_device()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        AIDE, DCTBaseRecModule = _load_aide_components(self.aide_root)
        model = AIDE(resnet_path=None, convnext_path=None)
        _patch_aide_forward_for_inference(model)

        checkpoint = torch.load(weight_path, map_location="cpu")
        state_dict = _strip_module_prefix(_state_dict_from_checkpoint(checkpoint))
        incompatible = model.load_state_dict(state_dict, strict=False)
        self.missing_keys = list(incompatible.missing_keys)
        self.unexpected_keys = list(incompatible.unexpected_keys)

        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("class_to_idx"), dict):
            self.class_to_idx = dict(checkpoint["class_to_idx"])

        for parameter in model.parameters():
            parameter.requires_grad = False

        try:
            model.to(self.device)
        except RuntimeError as exc:
            if self.device.type == "cuda" and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                model.to(self.device)
            else:
                raise

        model.eval()
        model.openclip_convnext_xxl.eval()

        self.model = model
        self.dct_module = DCTBaseRecModule()
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
            image_tensor = self._build_image_tensor(image_path)
            with torch.inference_mode():
                logits = self.model(image_tensor)
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

    def _build_image_tensor(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        image = self.to_tensor(image)
        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct_module(image)
        views = [
            self.normalize(x_minmin),
            self.normalize(x_maxmax),
            self.normalize(x_minmin1),
            self.normalize(x_maxmax1),
            self.normalize(image),
        ]
        return torch.stack(views, dim=0).unsqueeze(0).to(self.device, non_blocking=self.device.type == "cuda")
