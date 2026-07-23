from pathlib import Path

from models.torchvision_binary_model import TorchvisionBinaryModel
from system.app_config import resource_path


DEFAULT_WEIGHT = resource_path("models/weights/efficientnet_b0_ai_detector.pth")


class EfficientNetB0Model(TorchvisionBinaryModel):
    def __init__(
        self,
        weight_path=DEFAULT_WEIGHT,
        device="cuda",
        model_key="efficientnet_b0",
        model_name="EfficientNet-B0",
    ):
        super().__init__(
            model_key=model_key,
            model_name=model_name,
            weight_path=Path(weight_path),
            device=device,
            architecture="efficientnet_b0",
        )
