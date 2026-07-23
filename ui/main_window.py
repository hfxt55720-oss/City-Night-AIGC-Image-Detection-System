import gc

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from models.model_registry import MODEL_SPECS, model_name
from system.app_config import load_config, normalize_device, resource_path, save_config
from system.monitor import SystemMonitor
from ui.pages.batch_page import BatchPage
from ui.pages.dataset_page import DatasetPage
from ui.pages.detect_page import DetectPage
from ui.pages.history_page import HistoryPage
from ui.pages.model_page import ModelPage
from ui.pages.settings_page import SettingsPage
from ui.pages.statistics_page import StatisticsPage
from ui.pages.training_page import TrainingPage


APP_NAME = "城市夜景AIGC图像检测系统"
APP_ICON_PATH = resource_path("data/img/tup_1.png")
CHECKBOX_CHECKED_ICON = resource_path("ui/assets/checkbox_checked_black.svg").as_posix()


class ModelLoadWorker(QObject):
    loaded = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)
    finished = pyqtSignal(str)

    def __init__(self, model_key: str, device: str):
        super().__init__()
        self.model_key = model_key
        self.device = device

    def run(self):
        try:
            model = self._create_model()
            model.load()
            self.loaded.emit(self.model_key, model)
        except Exception as exc:
            self.failed.emit(self.model_key, str(exc))
        finally:
            self.finished.emit(self.model_key)

    def _create_model(self):
        spec = MODEL_SPECS.get(self.model_key)
        if spec is None:
            raise ValueError(f"Unsupported model: {self.model_key}")

        architecture = spec.get("architecture", self.model_key)
        weight_path = spec["weight_path"]
        name = spec["name"]

        if architecture == "aide":
            from models.aide_model import AIDEModel

            return AIDEModel(weight_path=weight_path, device=self.device, model_key=self.model_key, model_name=name)
        if architecture == "resnet50":
            from models.resnet_model import ResNet50Model

            return ResNet50Model(weight_path=weight_path, device=self.device, model_key=self.model_key, model_name=name)
        if architecture == "efficientnet_b0":
            from models.efficientnet_model import EfficientNetB0Model

            return EfficientNetB0Model(weight_path=weight_path, device=self.device, model_key=self.model_key, model_name=name)
        raise ValueError(f"Unsupported model architecture: {architecture}")


class MainWindow(QMainWindow):
    NAV_ITEMS = (
        ("图像检测", DetectPage),
        ("批量检测", BatchPage),
        ("模型管理", ModelPage),
        ("自定义模型训练", TrainingPage),
        ("数据集构建", DatasetPage),
        ("数据统计", StatisticsPage),
        ("记录管理", HistoryPage),
        ("系统设置", SettingsPage),
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        if APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.resize(1280, 800)
        self.setMinimumSize(1080, 680)

        self.current_model_key = self._read_default_model()
        self.current_device = self._read_default_device()
        self.loaded_models = {key: None for key in MODEL_SPECS}
        self.model_errors = {}
        self.model_threads = {}
        self.model_loaders = {}
        self.nav_buttons: list[QPushButton] = []
        self.pages: list[QWidget] = []
        self.page_stack = QStackedWidget()
        self.system_monitor = SystemMonitor()
        self.system_status_label = QLabel("系统监控初始化中...")
        self.system_status_label.setObjectName("SystemStatus")
        self.monitor_timer = QTimer(self)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setObjectName("MainSplitter")
        main_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(self._build_sidebar())
        main_splitter.addWidget(self._build_content())
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([224, 1056])
        root_layout.addWidget(main_splitter, 1)

        self.setCentralWidget(root)
        self._apply_styles()
        self._setup_status_bar()
        self._model_page().model_selected.connect(self.set_current_model)
        self._model_page().model_added.connect(self._on_model_added)
        self._detect_page().model_selected.connect(self.set_current_model)
        self._detect_page().device_selected.connect(self.set_current_device)
        self._settings_page().settings_saved.connect(self._on_settings_saved)
        self._model_page().set_current_model(self.current_model_key)
        self._detect_page().set_current_device(self.current_device)
        self._settings_page().set_current_model(self.current_model_key)
        self._settings_page().set_current_device(self.current_device)
        self._apply_current_model_to_detection_pages()
        self.switch_page(0)
        self._ensure_model_loaded(self.current_model_key)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setMinimumWidth(184)
        sidebar.setMaximumWidth(320)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 22, 16, 18)
        layout.setSpacing(10)

        title = QLabel("城市夜景\nAIGC检测")
        title.setObjectName("SidebarTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        title.setMinimumHeight(62)

        layout.addWidget(title)
        layout.addSpacing(18)

        for index, (label, _) in enumerate(self.NAV_ITEMS):
            button = QPushButton(label)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda checked=False, page_index=index: self.switch_page(page_index))
            self.nav_buttons.append(button)
            layout.addWidget(button)

        layout.addStretch()
        return sidebar

    def _build_content(self) -> QWidget:
        content = QWidget()
        content.setObjectName("Content")

        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        for _, page_class in self.NAV_ITEMS:
            page = page_class()
            self.pages.append(page)
            self.page_stack.addWidget(page)

        layout.addWidget(self.page_stack)
        return content

    def switch_page(self, index: int):
        self.page_stack.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        current_page = self.page_stack.currentWidget()
        if hasattr(current_page, "load_records"):
            current_page.load_records()

    def set_current_model(self, model_key: str):
        if model_key not in MODEL_SPECS:
            return

        self._sync_model_runtime_slots()
        self.current_model_key = model_key
        self._write_default_model(model_key)
        self._model_page().set_current_model(model_key)
        self._settings_page().set_current_model(model_key)
        self._apply_current_model_to_detection_pages()
        self._ensure_model_loaded(model_key)
        self.statusBar().showMessage(f"当前使用模型：{model_name(model_key)}")

    def _on_model_added(self, model_key: str):
        self._sync_model_runtime_slots()
        self._detect_page().refresh_model_options(self.current_model_key)
        self._settings_page().refresh_model_options(self.current_model_key)
        self.statusBar().showMessage(f"模型已添加：{model_name(model_key)}。点击“使用选中模型”后才会切换。")

    def set_current_device(self, device: str):
        normalized_device = self._normalize_device(device)
        if normalized_device == "cuda" and not self._cuda_available():
            self._detect_page().set_current_device(self.current_device)
            self._settings_page().set_current_device(self.current_device)
            self.statusBar().showMessage("当前环境未检测到可用CUDA，仍使用原设备。")
            return

        if normalized_device == self.current_device:
            self._detect_page().set_current_device(self.current_device)
            self._settings_page().set_current_device(self.current_device)
            return

        if self.model_threads:
            self._detect_page().set_current_device(self.current_device)
            self._settings_page().set_current_device(self.current_device)
            self.statusBar().showMessage("模型正在加载，加载完成后再切换CPU/GPU。")
            return

        self.current_device = normalized_device
        self._write_default_device(normalized_device)
        self._unload_loaded_models()
        self._detect_page().set_current_device(normalized_device)
        self._settings_page().set_current_device(normalized_device)
        self._apply_current_model_to_detection_pages()
        self._ensure_model_loaded(self.current_model_key)

        device_name = "GPU (CUDA)" if normalized_device == "cuda" else "CPU"
        self.statusBar().showMessage(f"运行设备已切换为：{device_name}")

    def _ensure_model_loaded(self, model_key: str):
        self._sync_model_runtime_slots()
        if self.loaded_models.get(model_key) is not None:
            self._apply_current_model_to_detection_pages()
            return
        if model_key in self.model_threads:
            return

        name = model_name(model_key)
        self._model_page().set_model_status(model_key, "加载中")
        if model_key == self.current_model_key:
            message = f"{name}模型正在加载，加载完成后即可检测。"
            self._detect_page().set_model_unavailable(name, message, model_key)
            self._batch_page().set_model_unavailable(name, message)
        self.statusBar().showMessage(f"{name}模型加载中...")

        thread = QThread(self)
        loader = ModelLoadWorker(model_key, self.current_device)
        loader.moveToThread(thread)

        thread.started.connect(loader.run)
        loader.loaded.connect(self._on_model_loaded)
        loader.failed.connect(self._on_model_failed)
        loader.finished.connect(thread.quit)
        loader.finished.connect(loader.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda key=model_key: self._clear_model_loader(key))

        self.model_threads[model_key] = thread
        self.model_loaders[model_key] = loader
        thread.start()

    def _on_model_loaded(self, model_key: str, model):
        self.loaded_models[model_key] = model
        self.model_errors.pop(model_key, None)
        self._model_page().set_model_status(model_key, "已加载")
        if model_key == self.current_model_key:
            self._apply_current_model_to_detection_pages()
        self.statusBar().showMessage(
            f"{model_name(model_key)}模型已加载 | device={model.device} | load={model.load_time:.1f}s"
        )

    def _on_model_failed(self, model_key: str, message: str):
        self.model_errors[model_key] = message
        self._model_page().set_model_status(model_key, "加载失败")
        if model_key == self.current_model_key:
            self._apply_current_model_to_detection_pages()
        short_message = " ".join(message.split())[:180]
        self.statusBar().showMessage(f"{model_name(model_key)}模型加载失败：{short_message}")

    def _clear_model_loader(self, model_key: str):
        self.model_threads.pop(model_key, None)
        self.model_loaders.pop(model_key, None)

    def _apply_current_model_to_detection_pages(self):
        self._detect_page().set_current_device(self.current_device)
        name = model_name(self.current_model_key)
        model = self.loaded_models.get(self.current_model_key)

        if model is not None:
            status = f"{name}模型已加载 | device={model.device}"
            self._detect_page().set_active_model(model, name, status, self.current_model_key)
            self._batch_page().set_active_model(model, name, status)
            return

        error = self.model_errors.get(self.current_model_key)
        if error:
            message = f"{name}模型加载失败：{error}"
        else:
            message = f"{name}模型尚未加载。"
        self._detect_page().set_model_unavailable(name, message, self.current_model_key)
        self._batch_page().set_model_unavailable(name, message)

    def _setup_status_bar(self):
        self.statusBar().setObjectName("MainStatusBar")
        self.system_status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.system_status_label.setMinimumWidth(760)
        self.statusBar().addPermanentWidget(self.system_status_label, 1)
        self.monitor_timer.timeout.connect(self._update_system_status)
        self.monitor_timer.start(1000)
        self._update_system_status()

    def _update_system_status(self):
        try:
            snapshot = self.system_monitor.snapshot()
            self.system_status_label.setText(snapshot.to_status_text())
        except Exception as exc:
            self.system_status_label.setText(f"系统监控不可用：{exc}")

    def _read_default_device(self) -> str:
        config = self._read_config()
        return self._normalize_device(config.get("device", "cuda"))

    def _read_default_model(self) -> str:
        config = self._read_config()
        model_key = config.get("default_model", "aide")
        return model_key if model_key in MODEL_SPECS else "aide"

    def _read_config(self) -> dict:
        return load_config()

    def _write_default_model(self, model_key: str):
        config = load_config()
        config["default_model"] = model_key
        save_config(config)

    def _write_default_device(self, device: str):
        config = load_config()
        config["device"] = self._normalize_device(device)
        save_config(config)

    def _normalize_device(self, device: str) -> str:
        return normalize_device(device)

    def _on_settings_saved(self, config: dict):
        self._detect_page().apply_settings(config)
        self._batch_page().apply_settings(config)

        status_message = "系统设置已保存并应用。"
        requested_device = self._normalize_device(config.get("device", self.current_device))
        if requested_device != self.current_device:
            if requested_device == "cuda" and not self._cuda_available():
                config["device"] = self.current_device
                save_config(config)
                self._settings_page().set_current_device(self.current_device)
                status_message = "设置已保存，但当前环境没有可用CUDA，运行设备保持不变。"
            elif self.model_threads:
                config["device"] = self.current_device
                save_config(config)
                self._settings_page().set_current_device(self.current_device)
                status_message = "设置已保存，但模型正在加载，运行设备保持不变。"
            else:
                self.set_current_device(requested_device)
        else:
            self._detect_page().set_current_device(self.current_device)
            self._settings_page().set_current_device(self.current_device)

        requested_model = config.get("default_model", self.current_model_key)
        if requested_model in MODEL_SPECS and requested_model != self.current_model_key:
            self.set_current_model(requested_model)
        else:
            self._settings_page().set_current_model(self.current_model_key)

        if self.page_stack.currentWidget() is self._history_page():
            self._history_page().load_records()
        if self.page_stack.currentWidget() is self._statistics_page():
            self._statistics_page().refresh_statistics()
        self.statusBar().showMessage(status_message)

    def _cuda_available(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def _unload_loaded_models(self):
        self._sync_model_runtime_slots()
        for key in MODEL_SPECS:
            self.loaded_models[key] = None
            self._model_page().set_model_status(key, "未加载")
        self.model_errors.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _sync_model_runtime_slots(self):
        for key in MODEL_SPECS:
            self.loaded_models.setdefault(key, None)

    def _detect_page(self) -> DetectPage:
        return self.pages[0]

    def _batch_page(self) -> BatchPage:
        return self.pages[1]

    def _model_page(self) -> ModelPage:
        return self.pages[2]

    def _statistics_page(self) -> StatisticsPage:
        return self.pages[5]

    def _history_page(self) -> HistoryPage:
        return self.pages[6]

    def _settings_page(self) -> SettingsPage:
        return self.pages[7]

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: #07090d;
            }
            QWidget {
                color: #e8edf3;
            }
            #Sidebar {
                background: #050608;
                border-right: 1px solid #171d25;
            }
            #SidebarTitle {
                color: #ffffff;
                font-size: 22px;
                font-weight: 700;
                line-height: 1.2;
            }
            #SidebarSubtitle {
                color: #8f9bad;
                font-size: 12px;
            }
            #SidebarFooter {
                color: #7f8b9d;
                font-size: 12px;
                padding: 8px 4px;
            }
            QPushButton {
                outline: none;
            }
            QPushButton#NavButton {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                color: #d7dee8;
                font-size: 15px;
                font-weight: 500;
                min-height: 42px;
                padding: 0 14px;
                text-align: left;
            }
            QPushButton#NavButton:hover {
                background: #151c24;
                border: 1px solid #2a3442;
                color: #ffffff;
            }
            QPushButton#NavButton:pressed {
                background: #0b1118;
                border: 1px solid #3b82f6;
                color: #ffffff;
                padding-left: 16px;
                padding-top: 1px;
            }
            QPushButton#NavButton:checked {
                background: #2563eb;
                border: 1px solid #60a5fa;
                color: #ffffff;
            }
            QPushButton#NavButton:checked:hover {
                background: #3b82f6;
                border: 1px solid #93c5fd;
            }
            QPushButton#NavButton:checked:pressed {
                background: #1d4ed8;
                border: 1px solid #bfdbfe;
                padding-left: 16px;
                padding-top: 1px;
            }
            #Content {
                background: #0b0f14;
            }
            QSplitter::handle {
                background: #202833;
            }
            QSplitter::handle:horizontal {
                width: 6px;
            }
            QSplitter::handle:vertical {
                height: 6px;
            }
            QSplitter::handle:hover {
                background: #3b4a5d;
            }
            QSplitter::handle:pressed {
                background: #3b82f6;
            }
            QSplitter#MainSplitter::handle {
                background: #171d25;
            }
            QSplitter#MainSplitter::handle:hover {
                background: #3b82f6;
            }
            #PageTitle {
                color: #f8fafc;
                font-size: 26px;
                font-weight: 700;
            }
            #PageSubtitle {
                color: #9aa6b2;
                font-size: 13px;
            }
            #Panel, #MetricCard {
                background: #111821;
                border: 1px solid #283241;
                border-radius: 8px;
            }
            #PanelTitle {
                color: #f1f5f9;
                font-size: 16px;
                font-weight: 700;
            }
            #MetricTitle {
                color: #9aa6b2;
                font-size: 13px;
            }
            #MetricValue {
                color: #f8fafc;
                font-size: 24px;
                font-weight: 700;
            }
            #MutedText {
                color: #9aa6b2;
                font-size: 12px;
            }
            #EmptyState {
                background: #0f151d;
                border: 1px dashed #354254;
                border-radius: 8px;
            }
            #EmptyStateText {
                color: #9aa6b2;
                font-size: 14px;
            }
            QLabel#FieldLabel {
                color: #a6b1bf;
                font-size: 13px;
            }
            QLabel#SelectorLabel {
                color: #e8edf3;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel#TrainingPromptLabel,
            QLabel#DatasetPromptLabel,
            QLabel#SettingsPromptLabel {
                color: #e8edf3;
                font-size: 13px;
                font-weight: 600;
                padding-right: 4px;
            }
            QLabel#TrainingSectionLabel,
            QLabel#DatasetSectionLabel,
            QLabel#SettingsSectionLabel {
                color: #60a5fa;
                font-size: 13px;
                font-weight: 700;
                padding: 12px 0 4px 0;
                border-bottom: 1px solid #2a3340;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #151c24;
                border: 1px solid #313b49;
                border-radius: 6px;
                color: #f1f5f9;
                min-height: 34px;
                padding: 0 10px;
            }
            QTextEdit, QPlainTextEdit {
                background: #0f151d;
                border: 1px solid #313b49;
                border-radius: 6px;
                color: #f1f5f9;
                font-size: 13px;
                padding: 8px;
            }
            QScrollBar:vertical {
                background: #0f151d;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 6px;
                min-height: 28px;
            }
            QScrollBar::handle:vertical:hover {
                background: #475569;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar:horizontal {
                background: #0f151d;
                height: 12px;
                margin: 0;
            }
            QScrollBar::handle:horizontal {
                background: #334155;
                border-radius: 6px;
                min-width: 28px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #475569;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0;
            }
            QComboBox#ModelSelector, QComboBox#DeviceSelector {
                background: #151c24;
                color: #f1f5f9;
                min-height: 38px;
                font-size: 14px;
            }
            QComboBox#ModelSelector QAbstractItemView,
            QComboBox#DeviceSelector QAbstractItemView,
            QComboBox#HistoryModelFilter QAbstractItemView {
                background: #151c24;
                color: #f1f5f9;
                selection-background-color: #243b5c;
                selection-color: #ffffff;
            }
            QComboBox#HistoryModelFilter {
                background: #151c24;
                color: #f1f5f9;
                min-height: 38px;
                min-width: 150px;
                font-size: 14px;
            }
            QPushButton#PrimaryButton {
                background: #2563eb;
                border: 1px solid #3b82f6;
                border-radius: 6px;
                color: #ffffff;
                font-weight: 600;
                min-height: 36px;
                padding: 0 16px;
            }
            QPushButton#PrimaryButton:hover {
                background: #3b82f6;
                border: 1px solid #60a5fa;
            }
            QPushButton#PrimaryButton:pressed {
                background: #1d4ed8;
                border: 1px solid #1e40af;
                padding: 1px 15px 0 17px;
            }
            QPushButton#SecondaryButton {
                background: #1a2330;
                border: 1px solid #313b49;
                border-radius: 6px;
                color: #e8edf3;
                min-height: 36px;
                padding: 0 16px;
            }
            QPushButton#SecondaryButton:hover {
                background: #243041;
                border: 1px solid #4b5b70;
                color: #ffffff;
            }
            QPushButton#SecondaryButton:pressed {
                background: #101821;
                border: 1px solid #60a5fa;
                color: #ffffff;
                padding: 1px 15px 0 17px;
            }
            QPushButton#TableActionButton {
                background: #1a2330;
                border: 1px solid #313b49;
                border-radius: 5px;
                color: #e8edf3;
                min-height: 28px;
                max-height: 30px;
                padding: 0 10px;
                font-size: 12px;
            }
            QPushButton#TableActionButton:hover {
                background: #243041;
                border: 1px solid #4b5b70;
                color: #ffffff;
            }
            QPushButton#TableActionButton:pressed {
                background: #101821;
                border: 1px solid #60a5fa;
                color: #ffffff;
                padding: 1px 9px 0 11px;
            }
            QPushButton#ModelCardButton {
                background: #111821;
                border: 2px solid #283241;
                border-radius: 8px;
                color: #dce3ec;
                font-size: 14px;
                font-weight: 600;
                min-height: 132px;
                padding: 14px 18px;
                text-align: left;
            }
            QPushButton#ModelCardButton:hover {
                border: 2px solid #60a5fa;
                background: #172233;
                color: #ffffff;
            }
            QPushButton#ModelCardButton:pressed {
                border: 2px solid #3b82f6;
                background: #101827;
                color: #ffffff;
                padding: 15px 17px 13px 19px;
            }
            QPushButton#ModelCardButton:checked {
                background: #172a46;
                border: 2px solid #3b82f6;
                color: #ffffff;
            }
            QPushButton#ModelCardButton:checked:hover {
                background: #1d3557;
                border: 2px solid #60a5fa;
            }
            QPushButton#ModelCardButton:checked:pressed {
                background: #13233a;
                border: 2px solid #93c5fd;
                padding: 15px 17px 13px 19px;
            }
            QPushButton#PrimaryButton:disabled,
            QPushButton#SecondaryButton:disabled,
            QPushButton#ModelCardButton:disabled {
                background: #151a21;
                border: 1px solid #252e3a;
                color: #667085;
            }
            QTableWidget {
                background: #10161d;
                alternate-background-color: #151c24;
                border: 1px solid #283241;
                border-radius: 6px;
                color: #e8edf3;
                gridline-color: #283241;
                selection-background-color: #243b5c;
                selection-color: #ffffff;
            }
            QWidget#HistoryActionCell,
            QWidget#HistorySelectionCell {
                background: transparent;
                border: none;
            }
            QHeaderView {
                background: #151d26;
            }
            QHeaderView::section {
                background: #151d26;
                border: none;
                border-bottom: 1px solid #283241;
                color: #c5ceda;
                font-weight: 600;
                min-height: 34px;
                padding: 6px;
            }
            QTableCornerButton::section {
                background: #151d26;
                border: none;
                border-bottom: 1px solid #283241;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox {
                color: #e8edf3;
                font-size: 13px;
                spacing: 8px;
            }
            QCheckBox::indicator:unchecked {
                background: #ffffff;
                border: 1px solid #9fb4ca;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked:hover {
                border: 1px solid #60a5fa;
                background: #e8edf3;
            }
            QCheckBox::indicator:checked {
                image: url(__CHECKBOX_CHECKED_ICON__);
                background: #ffffff;
                border: 1px solid #111820;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked:hover {
                background: #e8edf3;
                border: 1px solid #111820;
            }
            QCheckBox::indicator:checked:disabled {
                background: #9aa6b2;
                border: 1px solid #667085;
            }
            QProgressBar {
                background: #1a2330;
                border: none;
                border-radius: 6px;
                height: 12px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #3b82f6;
                border-radius: 6px;
            }
            QLabel#ImagePreview {
                background: #050608;
                border: 1px solid #283241;
                border-radius: 8px;
                color: #9aa6b2;
                font-size: 14px;
            }
            #ImageSlot {
                background: #0f151d;
                border: 1px solid #283241;
                border-radius: 8px;
            }
            QLabel#CamImagePreview {
                background: #050608;
                border: 1px solid transparent;
                border-radius: 6px;
                color: #9aa6b2;
                font-size: 13px;
            }
            QLabel#CamImagePreview:hover {
                border: 1px solid #60a5fa;
            }
            QLabel#HistoryImagePreview {
                background: #050608;
                border: 1px solid transparent;
                border-radius: 6px;
                color: #9aa6b2;
                font-size: 13px;
            }
            QLabel#HistoryImagePreview:hover {
                border: 1px solid #60a5fa;
            }
            QScrollArea#HistoryDetailScroll {
                background: transparent;
                border: none;
            }
            QScrollArea#TrainingConfigScroll,
            QScrollArea#DatasetConfigScroll {
                background: transparent;
                border: none;
            }
            QWidget#HistoryDetailContent {
                background: transparent;
            }
            QWidget#TrainingConfigContent,
            QWidget#DatasetConfigContent,
            QWidget#DatasetFormRow,
            QWidget#DatasetFieldBox {
                background: transparent;
            }
            #ResultBox {
                background: #0f151d;
                border: 1px solid #283241;
                border-radius: 8px;
            }
            QLabel#ResultValue {
                color: #f8fafc;
                font-size: 15px;
                font-weight: 700;
            }
            QStatusBar#MainStatusBar {
                background: #0b0f14;
                border-top: 1px solid #283241;
                color: #9aa6b2;
            }
            QLabel#SystemStatus {
                color: #9aa6b2;
                font-size: 12px;
                padding-right: 8px;
            }
            """.replace("__CHECKBOX_CHECKED_ICON__", CHECKBOX_CHECKED_ICON)
        )
