import json
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QEvent, QObject, QPoint, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from models.model_registry import MODEL_SPECS, model_name
from system.app_config import behavior_enabled, load_config, path_for
from system.history_store import append_history
from ui.pages.common import page_header, page_root, panel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "output" / "results"
DEFAULT_HEATMAP_DIR = PROJECT_ROOT / "output" / "heatmaps"


class PredictionWorker(QObject):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, model, image_path: str):
        super().__init__()
        self.model = model
        self.image_path = image_path

    def run(self):
        try:
            self.finished.emit(self.model.predict(self.image_path))
        except Exception as exc:
            self.failed.emit(str(exc))


class GradCAMWorker(QObject):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, model, image_path: str, output_dir: Path):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.output_dir = Path(output_dir)

    def run(self):
        try:
            from inference.gradcam import generate_gradcam

            self.finished.emit(generate_gradcam(self.model, self.image_path, self.output_dir))
        except Exception as exc:
            self.failed.emit(str(exc))


class ImagePreviewLabel(QLabel):
    clicked = pyqtSignal(str)

    def __init__(self, image_key: str, empty_text: str):
        super().__init__(empty_text)
        self.image_key = image_key
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("点击放大查看")

    def mousePressEvent(self, event):
        pixmap = self.pixmap()
        if event.button() == Qt.MouseButton.LeftButton and pixmap is not None and not pixmap.isNull():
            self.clicked.emit(self.image_key)
            return
        super().mousePressEvent(event)


class ImageViewerDialog(QDialog):
    def __init__(self, image_path: Path, title: str, parent=None):
        super().__init__(parent)
        self.image_path = Path(image_path)
        self.base_pixmap = QPixmap(str(self.image_path))
        self.scale_factor = 1.0
        self.dragging = False
        self.last_drag_pos = QPoint()

        self.setWindowTitle(title)
        self.resize(920, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title_label = QLabel(f"{title} | {self.image_path.name}")
        title_label.setObjectName("PanelTitle")
        title_label.setWordWrap(True)

        self.image_label = QLabel("图片无法打开")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setBackgroundRole(self.backgroundRole())

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        self.scroll_area.viewport().installEventFilter(self)
        self.image_label.installEventFilter(self)

        button_row = QHBoxLayout()
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("MutedText")

        zoom_out_button = QPushButton("缩小")
        zoom_out_button.setObjectName("SecondaryButton")
        zoom_out_button.clicked.connect(self.zoom_out)

        zoom_in_button = QPushButton("放大")
        zoom_in_button.setObjectName("SecondaryButton")
        zoom_in_button.clicked.connect(self.zoom_in)

        reset_button = QPushButton("100%")
        reset_button.setObjectName("SecondaryButton")
        reset_button.clicked.connect(self.reset_zoom)

        fit_button = QPushButton("适应窗口")
        fit_button.setObjectName("PrimaryButton")
        fit_button.clicked.connect(self.fit_to_window)

        button_row.addWidget(zoom_out_button)
        button_row.addWidget(zoom_in_button)
        button_row.addWidget(reset_button)
        button_row.addWidget(fit_button)
        button_row.addStretch()
        button_row.addWidget(self.zoom_label)

        layout.addWidget(title_label)
        layout.addWidget(self.scroll_area, 1)
        layout.addLayout(button_row)

        self._apply_scaled_pixmap()
        QTimer.singleShot(0, self.fit_to_window)

    def eventFilter(self, source, event):
        if source not in (self.scroll_area.viewport(), self.image_label):
            return super().eventFilter(source, event)

        if event.type() == QEvent.Type.Wheel:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            return True

        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_drag_pos = self._global_pos(event)
            self.scroll_area.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return True

        if event.type() == QEvent.Type.MouseMove and self.dragging:
            current_pos = self._global_pos(event)
            delta = current_pos - self.last_drag_pos
            self.last_drag_pos = current_pos
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - delta.x()
            )
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - delta.y()
            )
            return True

        if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.scroll_area.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return True

        return super().eventFilter(source, event)

    def zoom_in(self):
        self._set_scale(self.scale_factor * 1.2)

    def zoom_out(self):
        self._set_scale(self.scale_factor / 1.2)

    def reset_zoom(self):
        self._set_scale(1.0)

    def fit_to_window(self):
        if self.base_pixmap.isNull():
            return
        viewport_size = self.scroll_area.viewport().size()
        if viewport_size.width() <= 0 or viewport_size.height() <= 0:
            return
        width_scale = (viewport_size.width() - 8) / self.base_pixmap.width()
        height_scale = (viewport_size.height() - 8) / self.base_pixmap.height()
        self._set_scale(min(width_scale, height_scale))

    def _set_scale(self, value: float):
        self.scale_factor = min(max(value, 0.05), 8.0)
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self):
        if self.base_pixmap.isNull():
            self.zoom_label.setText("-")
            return

        width = max(1, int(self.base_pixmap.width() * self.scale_factor))
        height = max(1, int(self.base_pixmap.height() * self.scale_factor))
        scaled = self.base_pixmap.scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())
        self.zoom_label.setText(f"{int(self.scale_factor * 100)}%")

    def _global_pos(self, event) -> QPoint:
        if hasattr(event, "globalPosition"):
            return event.globalPosition().toPoint()
        return event.globalPos()


class DetectPage(QWidget):
    model_selected = pyqtSignal(str)
    device_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.active_model = None
        self.current_model_key = "aide"
        self.current_device = "cuda"
        self.active_model_name = "AIDE-Night"
        self.current_image_path: Path | None = None
        self.current_result: dict | None = None
        self.current_gradcam: dict | None = None
        self.current_save_signature: str | None = None
        self.pending_prediction_after_model_switch = False
        self.visual_pixmaps: dict[str, QPixmap] = {}
        self.visual_paths: dict[str, Path] = {}
        self.prediction_thread: QThread | None = None
        self.prediction_worker: PredictionWorker | None = None
        self.gradcam_thread: QThread | None = None
        self.gradcam_worker: GradCAMWorker | None = None
        self.apply_settings(load_config())

        layout = page_root(self)
        layout.addWidget(page_header("图像检测", "上传单张城市夜景图片，检测后自动生成 Grad-CAM 可视化结果。"))

        workspace = QSplitter(Qt.Orientation.Horizontal)
        workspace.setChildrenCollapsible(False)
        workspace.setHandleWidth(8)

        visual_panel, visual_layout = panel("可视化结果")
        self.original_label = self._build_visual_slot(visual_layout, "原图", "未选择图片", minimum_height=280)

        cam_splitter = QSplitter(Qt.Orientation.Horizontal)
        cam_splitter.setChildrenCollapsible(False)
        cam_splitter.setHandleWidth(8)
        self.heatmap_label = self._build_visual_slot(cam_splitter, "Grad-CAM热力图", "等待生成", minimum_height=220)
        self.overlay_label = self._build_visual_slot(cam_splitter, "叠加图", "等待生成", minimum_height=220)
        cam_splitter.setSizes([1, 1])
        visual_layout.addWidget(cam_splitter)

        result_panel, result_layout = panel("检测结果")
        result_panel.setMinimumWidth(320)
        result_panel.setMaximumWidth(560)

        self.model_status_label = QLabel("AIDE-Night模型加载中...")
        self.model_status_label.setObjectName("MutedText")
        self.model_status_label.setWordWrap(True)

        self.upload_button = QPushButton("上传图片")
        self.upload_button.setObjectName("PrimaryButton")
        self.upload_button.clicked.connect(self.upload_image)

        self.save_button = QPushButton("保存检测结果")
        self.save_button.setObjectName("SecondaryButton")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_result)

        action_row = QHBoxLayout()
        action_row.addWidget(self.upload_button)
        action_row.addWidget(self.save_button)

        model_label = QLabel("检测模型")
        model_label.setObjectName("SelectorLabel")
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("ModelSelector")
        for key, spec in MODEL_SPECS.items():
            self.model_combo.addItem(spec["name"], key)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        self._sync_model_combo()

        device_label = QLabel("运行设备")
        device_label.setObjectName("SelectorLabel")
        self.device_combo = QComboBox()
        self.device_combo.setObjectName("DeviceSelector")
        self.device_combo.addItem("GPU (CUDA)", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.currentIndexChanged.connect(self._on_device_combo_changed)
        self._sync_device_combo()

        self.result_labels = {}
        result_box = QFrame()
        result_box.setObjectName("ResultBox")
        result_grid = QGridLayout(result_box)
        result_grid.setContentsMargins(16, 16, 16, 16)
        result_grid.setHorizontalSpacing(12)
        result_grid.setVerticalSpacing(14)

        fields = [
            ("类别", "result"),
            ("AIGC概率", "ai_probability"),
            ("REAL概率", "real_probability"),
            ("模型名称", "model_name"),
            ("推理时间", "time"),
            ("热力图", "gradcam"),
        ]
        for row, (label_text, key) in enumerate(fields):
            name_label = QLabel(label_text)
            name_label.setObjectName("FieldLabel")
            value_label = QLabel("-")
            value_label.setObjectName("ResultValue")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_label.setWordWrap(True)
            result_grid.addWidget(name_label, row, 0)
            result_grid.addWidget(value_label, row, 1)
            self.result_labels[key] = value_label

        self.message_label = QLabel("等待上传图片")
        self.message_label.setObjectName("MutedText")
        self.message_label.setWordWrap(True)

        result_layout.addWidget(self.model_status_label)
        result_layout.addLayout(action_row)
        result_layout.addWidget(model_label)
        result_layout.addWidget(self.model_combo)
        result_layout.addWidget(device_label)
        result_layout.addWidget(self.device_combo)
        result_layout.addWidget(result_box)
        result_layout.addWidget(self.message_label)
        result_layout.addStretch()

        workspace.addWidget(visual_panel)
        workspace.addWidget(result_panel)
        workspace.setStretchFactor(0, 1)
        workspace.setStretchFactor(1, 0)
        workspace.setSizes([820, 380])
        layout.addWidget(workspace, 1)

    def set_aide_model(self, model):
        self.set_active_model(model, "AIDE-Night", f"AIDE-Night模型已加载 | device={model.device}", "aide")

    def set_active_model(self, model, model_name: str, status_message: str | None = None, model_key: str | None = None):
        resolved_key = model_key or getattr(model, "model_key", None) or self._model_key_from_name(model_name)
        if resolved_key in MODEL_SPECS:
            self.current_model_key = resolved_key
            self.refresh_model_options(self.current_model_key)
            self._sync_model_combo()

        self.active_model = model
        self.active_model_name = model_name
        self.model_status_label.setText(status_message or f"当前模型：{model_name}")
        self.message_label.setText("可以上传图片进行检测" if model is not None else f"{model_name}模型尚未加载")
        self.result_labels["model_name"].setText(model_name)
        self.model_combo.setEnabled(not self._is_busy())
        self.device_combo.setEnabled(not self._is_busy())

        if (
            self.pending_prediction_after_model_switch
            and model is not None
            and self.current_image_path is not None
            and not self._is_busy()
        ):
            self.pending_prediction_after_model_switch = False
            self.current_result = None
            self.current_gradcam = None
            self.current_save_signature = None
            self.save_button.setEnabled(False)
            self._reset_result_labels()
            self.result_labels["model_name"].setText(model_name)
            self.result_labels["gradcam"].setText("-")
            self._clear_gradcam_images()
            QTimer.singleShot(0, self._start_prediction)

    def set_model_unavailable(self, model_name: str, message: str, model_key: str | None = None):
        resolved_key = model_key or self._model_key_from_name(model_name)
        if resolved_key in MODEL_SPECS:
            self.current_model_key = resolved_key
            self.refresh_model_options(self.current_model_key)
            self._sync_model_combo()
        self.active_model = None
        self.active_model_name = model_name
        self.model_status_label.setText(f"当前模型：{model_name}")
        self.message_label.setText(message)
        self.result_labels["model_name"].setText(model_name)
        self.model_combo.setEnabled(not self._is_busy())
        self.device_combo.setEnabled(not self._is_busy())

    def set_model_error(self, message: str):
        self.active_model = None
        self.current_model_key = "aide"
        self.active_model_name = "AIDE-Night"
        self.refresh_model_options(self.current_model_key)
        self._sync_model_combo()
        self.model_status_label.setText(f"AIDE-Night模型加载失败：{message}")
        self.message_label.setText("请检查AIDE权重、运行环境和显存状态。")

    def set_current_device(self, device: str):
        normalized = "cpu" if str(device).lower().startswith("cpu") else "cuda"
        self.current_device = normalized
        self._sync_device_combo()

    def apply_settings(self, config: dict):
        self.result_dir = path_for("results", config)
        self.heatmap_dir = path_for("heatmaps", config)
        self.save_history_enabled = behavior_enabled("save_history", config)
        self.auto_gradcam_enabled = behavior_enabled("auto_gradcam", config)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        if not file_path:
            return

        self.current_image_path = Path(file_path)
        self.current_result = None
        self.current_gradcam = None
        self.current_save_signature = None
        self.pending_prediction_after_model_switch = False
        self.save_button.setEnabled(False)
        self._reset_result_labels()
        self.result_labels["model_name"].setText(self.active_model_name)
        self.result_labels["gradcam"].setText("-")
        self._clear_gradcam_images()
        self._show_image(self.current_image_path, self.original_label, "original")
        self._start_prediction()

    def save_result(self):
        if self.current_result is None or self.current_image_path is None:
            QMessageBox.information(self, "保存检测结果", "当前没有可保存的检测结果。")
            return

        self.result_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.result_dir / f"single_{self.current_model_key}_results.json"
        payload = self._build_save_payload()
        save_signature = self._save_signature(payload)
        if self.current_save_signature == save_signature:
            reply = QMessageBox.question(
                self,
                "重复保存",
                "当前检测结果已经保存过，是否继续重复保存？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.message_label.setText("已取消重复保存。")
                return

        try:
            record_count = self._append_json_record(save_path, payload)
        except ValueError as exc:
            QMessageBox.warning(self, "保存检测结果", str(exc))
            return

        history_payload = dict(payload)
        history_payload["source"] = "single"
        history_payload["result_file"] = str(save_path)
        if self.save_history_enabled:
            append_history(history_payload)
        self.current_save_signature = save_signature
        history_text = "已写入记录管理" if self.save_history_enabled else "未写入记录管理"
        self.message_label.setText(f"检测结果已保存：{save_path}（共 {record_count} 条，{history_text}）")

    def _start_prediction(self):
        if self.active_model is None:
            self.message_label.setText(f"{self.active_model_name}模型不可用或仍在加载。")
            return
        if self.current_image_path is None:
            return

        self.upload_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.message_label.setText("正在检测...")
        self.prediction_thread = QThread(self)
        self.prediction_worker = PredictionWorker(self.active_model, str(self.current_image_path))
        self.prediction_worker.moveToThread(self.prediction_thread)

        self.prediction_thread.started.connect(self.prediction_worker.run)
        self.prediction_worker.finished.connect(self._on_prediction_finished)
        self.prediction_worker.failed.connect(self._on_prediction_failed)
        self.prediction_worker.finished.connect(self.prediction_thread.quit)
        self.prediction_worker.failed.connect(self.prediction_thread.quit)
        self.prediction_worker.finished.connect(self.prediction_worker.deleteLater)
        self.prediction_worker.failed.connect(self.prediction_worker.deleteLater)
        self.prediction_thread.finished.connect(self.prediction_thread.deleteLater)
        self.prediction_thread.finished.connect(self._clear_prediction_worker)
        self.prediction_thread.start()

    def _start_gradcam(self):
        if self.active_model is None or self.current_image_path is None:
            self.upload_button.setEnabled(True)
            self.save_button.setEnabled(self.current_result is not None)
            self.model_combo.setEnabled(True)
            self.device_combo.setEnabled(True)
            return

        self.result_labels["gradcam"].setText("生成中")
        self.message_label.setText("正在生成Grad-CAM热力图...")
        self.gradcam_thread = QThread(self)
        self.gradcam_worker = GradCAMWorker(self.active_model, str(self.current_image_path), self.heatmap_dir)
        self.gradcam_worker.moveToThread(self.gradcam_thread)

        self.gradcam_thread.started.connect(self.gradcam_worker.run)
        self.gradcam_worker.finished.connect(self._on_gradcam_finished)
        self.gradcam_worker.failed.connect(self._on_gradcam_failed)
        self.gradcam_worker.finished.connect(self.gradcam_thread.quit)
        self.gradcam_worker.failed.connect(self.gradcam_thread.quit)
        self.gradcam_worker.finished.connect(self.gradcam_worker.deleteLater)
        self.gradcam_worker.failed.connect(self.gradcam_worker.deleteLater)
        self.gradcam_thread.finished.connect(self.gradcam_thread.deleteLater)
        self.gradcam_thread.finished.connect(self._clear_gradcam_worker)
        self.gradcam_thread.start()

    def _on_prediction_finished(self, result: dict):
        self.current_result = {
            "result": result["result"],
            "ai_probability": result["ai_probability"],
            "real_probability": result["real_probability"],
            "time": result["time"],
        }
        self.current_save_signature = None
        self.result_labels["result"].setText(result["result"])
        self.result_labels["ai_probability"].setText(f"{result['ai_probability']:.6f}")
        self.result_labels["real_probability"].setText(f"{result['real_probability']:.6f}")
        self.result_labels["model_name"].setText(self.active_model_name)
        self.result_labels["time"].setText(result["time"])
        if self.auto_gradcam_enabled:
            self._start_gradcam()
            return

        self.current_gradcam = None
        self.result_labels["gradcam"].setText("未生成")
        self.save_button.setEnabled(True)
        self.upload_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.message_label.setText("检测完成，已按系统设置跳过Grad-CAM生成。")

    def _on_prediction_failed(self, message: str):
        self.current_result = None
        self.save_button.setEnabled(False)
        self.upload_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.message_label.setText(f"检测失败：{message}")

    def _on_gradcam_finished(self, result: dict):
        self.current_gradcam = result
        self._show_image(Path(result["original_path"]), self.original_label, "original")
        self._show_image(Path(result["heatmap_path"]), self.heatmap_label, "heatmap")
        self._show_image(Path(result["overlay_path"]), self.overlay_label, "overlay")
        self.result_labels["gradcam"].setText("已生成")
        self.save_button.setEnabled(True)
        self.upload_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.message_label.setText(f"检测完成，Grad-CAM已保存：{self.heatmap_dir}")

    def _on_gradcam_failed(self, message: str):
        self.current_gradcam = None
        self.result_labels["gradcam"].setText("生成失败")
        self.save_button.setEnabled(self.current_result is not None)
        self.upload_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.message_label.setText(f"检测完成，但Grad-CAM生成失败：{message}")

    def _clear_prediction_worker(self):
        self.prediction_thread = None
        self.prediction_worker = None

    def _clear_gradcam_worker(self):
        self.gradcam_thread = None
        self.gradcam_worker = None

    def _reset_result_labels(self):
        for label in self.result_labels.values():
            label.setText("-")

    def _clear_gradcam_images(self):
        for key, label in [("heatmap", self.heatmap_label), ("overlay", self.overlay_label)]:
            self.visual_pixmaps.pop(key, None)
            self.visual_paths.pop(key, None)
            label.clear()
            label.setText("等待生成")

    def _build_visual_slot(self, parent_layout, title: str, empty_text: str, minimum_height: int):
        wrapper = QFrame()
        wrapper.setObjectName("ImageSlot")
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(12, 12, 12, 12)
        wrapper_layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setObjectName("FieldLabel")
        image_label = ImagePreviewLabel(self._visual_key_from_title(title), empty_text)
        image_label.setObjectName("CamImagePreview")
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setMinimumHeight(minimum_height)
        image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        image_label.clicked.connect(self.open_visual_image)

        wrapper_layout.addWidget(title_label)
        wrapper_layout.addWidget(image_label, 1)
        parent_layout.addWidget(wrapper)
        return image_label

    def _show_image(self, image_path: Path, label: QLabel, key: str):
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.visual_pixmaps.pop(key, None)
            self.visual_paths.pop(key, None)
            label.clear()
            label.setText("图片无法预览")
            return
        self.visual_pixmaps[key] = pixmap
        self.visual_paths[key] = Path(image_path)
        self._update_visual_pixmap(key, label)

    def _update_visual_pixmap(self, key: str, label: QLabel):
        pixmap = self.visual_pixmaps.get(key)
        if pixmap is None:
            return
        scaled = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)

    def open_visual_image(self, key: str):
        image_path = self.visual_paths.get(key)
        if image_path is None or not image_path.exists():
            QMessageBox.information(self, "图片查看", "当前图片尚未生成或文件不存在。")
            return

        titles = {
            "original": "原图",
            "heatmap": "Grad-CAM热力图",
            "overlay": "Grad-CAM叠加图",
        }
        dialog = ImageViewerDialog(image_path, titles.get(key, "图片预览"), self)
        dialog.exec()

    def _on_model_combo_changed(self):
        model_key = self.model_combo.currentData()
        if not model_key or model_key == self.current_model_key:
            return
        if self._is_busy():
            self._sync_model_combo()
            QMessageBox.information(self, "模型切换", "当前检测尚未完成，请稍后再切换模型。")
            return

        self.current_model_key = model_key
        self.active_model = None
        self.active_model_name = model_name(model_key)
        self.pending_prediction_after_model_switch = self.current_image_path is not None
        self.current_result = None
        self.current_gradcam = None
        self.current_save_signature = None
        self.save_button.setEnabled(False)
        self._reset_result_labels()
        self.result_labels["model_name"].setText(self.active_model_name)
        self.result_labels["gradcam"].setText("-")
        self._clear_gradcam_images()
        self.model_status_label.setText(f"当前模型：{self.active_model_name}")
        self.message_label.setText(f"正在切换到 {self.active_model_name} 模型...")
        self.model_selected.emit(model_key)

    def _on_device_combo_changed(self):
        device = self.device_combo.currentData()
        if not device or device == self.current_device:
            return
        if self._is_busy():
            self._sync_device_combo()
            QMessageBox.information(self, "设备切换", "当前检测尚未完成，请稍后再切换设备。")
            return

        self.current_device = device
        self.active_model = None
        self.current_result = None
        self.current_gradcam = None
        self.current_save_signature = None
        self.pending_prediction_after_model_switch = self.current_image_path is not None
        self.save_button.setEnabled(False)
        self._reset_result_labels()
        self.result_labels["model_name"].setText(self.active_model_name)
        self.result_labels["gradcam"].setText("-")
        self._clear_gradcam_images()
        self.message_label.setText(f"正在切换到 {'GPU (CUDA)' if device == 'cuda' else 'CPU'} 运行...")
        self.device_selected.emit(device)

    def _sync_model_combo(self):
        index = self.model_combo.findData(self.current_model_key)
        if index < 0:
            self.refresh_model_options(self.current_model_key)
            index = self.model_combo.findData(self.current_model_key)
        if index < 0:
            return
        was_blocked = self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(was_blocked)

    def refresh_model_options(self, current_model_key: str | None = None):
        selected_key = current_model_key or self.current_model_key
        was_blocked = self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for key, spec in MODEL_SPECS.items():
            self.model_combo.addItem(spec["name"], key)
        index = self.model_combo.findData(selected_key)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(was_blocked)

    def _sync_device_combo(self):
        index = self.device_combo.findData(self.current_device)
        if index < 0:
            return
        was_blocked = self.device_combo.blockSignals(True)
        self.device_combo.setCurrentIndex(index)
        self.device_combo.blockSignals(was_blocked)

    def _append_json_record(self, save_path: Path, payload: dict) -> int:
        records = []
        if save_path.exists():
            try:
                with save_path.open("r", encoding="utf-8") as file:
                    existing = json.load(file)
            except json.JSONDecodeError as exc:
                raise ValueError(f"目标文件不是有效JSON，未写入：{save_path}") from exc

            if isinstance(existing, list):
                records = existing
            elif isinstance(existing, dict):
                if isinstance(existing.get("records"), list):
                    existing["records"].append(payload)
                    existing["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with save_path.open("w", encoding="utf-8") as file:
                        json.dump(existing, file, ensure_ascii=False, indent=2)
                    return len(existing["records"])
                records = [existing]
            else:
                raise ValueError(f"目标文件格式不支持，未写入：{save_path}")

        records.append(payload)
        with save_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
        return len(records)

    def _build_save_payload(self) -> dict:
        if self.current_result is None or self.current_image_path is None:
            return {}

        saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "timestamp": saved_at,
            "saved_at": saved_at,
            "image_name": self.current_image_path.name,
            "image_path": str(self.current_image_path),
            "model_key": self.current_model_key,
            "model_name": self.active_model_name,
            "device": str(getattr(self.active_model, "device", "")),
            **self.current_result,
        }
        if self.current_gradcam is not None:
            payload["gradcam"] = self.current_gradcam
        return payload

    def _save_signature(self, payload: dict) -> str:
        stable_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"timestamp", "saved_at"}
        }
        return json.dumps(stable_payload, ensure_ascii=False, sort_keys=True)

    def _model_key_from_name(self, name: str) -> str | None:
        for key, spec in MODEL_SPECS.items():
            if spec["name"] == name:
                return key
        return None

    def _visual_key_from_title(self, title: str) -> str:
        if "热力图" in title:
            return "heatmap"
        if "叠加" in title:
            return "overlay"
        return "original"

    def _is_busy(self) -> bool:
        return self.prediction_thread is not None or self.gradcam_thread is not None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_visual_pixmap("original", self.original_label)
        self._update_visual_pixmap("heatmap", self.heatmap_label)
        self._update_visual_pixmap("overlay", self.overlay_label)
