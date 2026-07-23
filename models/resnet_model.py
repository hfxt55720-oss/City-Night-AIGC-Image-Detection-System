from pathlib import Path

from models.torchvision_binary_model import TorchvisionBinaryModel
from system.app_config import resource_path


DEFAULT_WEIGHT = resource_path("models/weights/resnet50_ai_detector.pth")


class ResNet50Model(TorchvisionBinaryModel):
    def __init__(self, weight_path=DEFAULT_WEIGHT, device="cuda", model_key="resnet50", model_name="ResNet50"):
        super().__init__(
            model_key=model_key,
            model_name=model_name,
            weight_path=Path(weight_path),
            device=device,
            architecture="resnet50",
        )
