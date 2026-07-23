import platform
import warnings
from dataclasses import dataclass

import psutil

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pynvml
except Exception:
    pynvml = None


@dataclass
class SystemSnapshot:
    cpu_name: str
    cpu_percent: float
    gpu_name: str
    gpu_percent: float | None
    gpu_memory_used_gb: float | None
    gpu_memory_total_gb: float | None
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float

    def to_status_text(self) -> str:
        gpu_usage = "GPU: N/A"
        if self.gpu_percent is not None and self.gpu_memory_used_gb is not None and self.gpu_memory_total_gb is not None:
            gpu_usage = (
                f"GPU: {self.gpu_name} {self.gpu_percent:.0f}% | "
                f"显存 {self.gpu_memory_used_gb:.1f}/{self.gpu_memory_total_gb:.1f}GB"
            )

        return (
            f"CPU: {self.cpu_name} {self.cpu_percent:.0f}% | "
            f"{gpu_usage} | "
            f"内存 {self.memory_used_gb:.1f}/{self.memory_total_gb:.1f}GB ({self.memory_percent:.0f}%)"
        )


class SystemMonitor:
    def __init__(self, gpu_index: int = 0):
        self.cpu_name = get_cpu_name()
        self.gpu_index = gpu_index
        self.gpu_handle = None
        self.gpu_name = "未检测到NVIDIA GPU"
        self._init_gpu()
        psutil.cpu_percent(interval=None)

    def snapshot(self) -> SystemSnapshot:
        memory = psutil.virtual_memory()
        gpu_percent = None
        gpu_memory_used_gb = None
        gpu_memory_total_gb = None

        if self.gpu_handle is not None and pynvml is not None:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_percent = float(utilization.gpu)
                gpu_memory_used_gb = bytes_to_gb(gpu_memory.used)
                gpu_memory_total_gb = bytes_to_gb(gpu_memory.total)
            except Exception:
                gpu_percent = None
                gpu_memory_used_gb = None
                gpu_memory_total_gb = None

        return SystemSnapshot(
            cpu_name=self.cpu_name,
            cpu_percent=float(psutil.cpu_percent(interval=None)),
            gpu_name=self.gpu_name,
            gpu_percent=gpu_percent,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            memory_used_gb=bytes_to_gb(memory.used),
            memory_total_gb=bytes_to_gb(memory.total),
            memory_percent=float(memory.percent),
        )

    def _init_gpu(self):
        if pynvml is None:
            self.gpu_name = "pynvml未安装"
            return

        try:
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() <= self.gpu_index:
                return
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            raw_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
            self.gpu_name = raw_name.decode("utf-8", errors="ignore") if isinstance(raw_name, bytes) else str(raw_name)
        except Exception:
            self.gpu_handle = None
            self.gpu_name = "GPU信息不可用"


def get_cpu_name() -> str:
    if platform.system().lower() == "windows":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if name:
                    return " ".join(str(name).split())
        except Exception:
            pass

    processor = platform.processor() or platform.machine() or "Unknown CPU"
    return " ".join(processor.split())


def bytes_to_gb(value: int) -> float:
    return float(value) / 1024 / 1024 / 1024


def get_system_info():
    monitor = SystemMonitor()
    snapshot = monitor.snapshot()
    return {
        "cpu_name": snapshot.cpu_name,
        "cpu_percent": snapshot.cpu_percent,
        "gpu_name": snapshot.gpu_name,
        "gpu_percent": snapshot.gpu_percent,
        "gpu_memory_used_gb": snapshot.gpu_memory_used_gb,
        "gpu_memory_total_gb": snapshot.gpu_memory_total_gb,
        "memory_used_gb": snapshot.memory_used_gb,
        "memory_total_gb": snapshot.memory_total_gb,
        "memory_percent": snapshot.memory_percent,
    }
