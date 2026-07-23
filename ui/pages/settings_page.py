from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from models.model_registry import MODEL_SPECS
from system.app_config import CONFIG_PATH, default_config, display_path, load_config, path_for, resolve_path, save_config
from ui.pages.common import page_header, page_root, panel


class SettingsPage(QWidget):
    settings_saved = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        layout = page_root(self)
        layout.addWidget(page_header("系统设置", "配置默认模型、运行设备、输出路径和检测行为，保存后会同步到实际检测流程。"))

        workspace = QSplitter(Qt.Orientation.Horizontal)
        workspace.setChildrenCollapsible(False)
        workspace.setHandleWidth(8)

        runtime_panel, runtime_layout = panel("运行设置")
        runtime_layout.addLayout(self._build_runtime_form())
        runtime_layout.addStretch()

        path_panel, path_layout = panel("输出与记录")
        path_layout.addLayout(self._build_path_form())
        path_layout.addWidget(self._build_action_panel())
        path_layout.addStretch()

        workspace.addWidget(runtime_panel)
        workspace.addWidget(path_panel)
        workspace.setStretchFactor(0, 1)
        workspace.setStretchFactor(1, 1)
        workspace.setSizes([500, 620])
        layout.addWidget(workspace, 1)

        self.load_settings(load_config())

    def _build_runtime_form(self) -> QFormLayout:
        form = self._new_form()

        self.default_model_combo = QComboBox()
        self.refresh_model_options()

        self.device_combo = QComboBox()
        self.device_combo.addItem("GPU (CUDA)", "cuda")
        self.device_combo.addItem("CPU", "cpu")

        self.save_history_checkbox = QCheckBox("检测完成后写入记录管理")
        self.auto_gradcam_checkbox = QCheckBox("单张检测完成后自动生成 Grad-CAM")

        form.addRow(self._section_label("默认检测"))
        self._add_form_row(form, "默认模型", self.default_model_combo, "软件启动后默认使用的检测模型。")
        self._add_form_row(form, "运行设备", self.device_combo, "默认使用 GPU CUDA 或 CPU 进行模型推理。")

        form.addRow(self._section_label("行为设置"))
        self._add_form_row(form, "检测记录", self.save_history_checkbox, "关闭后检测结果只在当前页面显示，不写入记录管理。")
        self._add_form_row(form, "热力图", self.auto_gradcam_checkbox, "关闭后单张检测不会自动生成 Grad-CAM 图片。")
        return form

    def _build_path_form(self) -> QFormLayout:
        form = self._new_form()

        self.result_dir_edit = QLineEdit()
        self.heatmap_dir_edit = QLineEdit()
        self.history_path_edit = QLineEdit()

        self.result_dir_button = QPushButton("选择")
        self.result_dir_button.setObjectName("SecondaryButton")
        self.result_dir_button.clicked.connect(lambda: self._choose_directory(self.result_dir_edit, "选择结果保存目录"))

        self.heatmap_dir_button = QPushButton("选择")
        self.heatmap_dir_button.setObjectName("SecondaryButton")
        self.heatmap_dir_button.clicked.connect(lambda: self._choose_directory(self.heatmap_dir_edit, "选择热力图保存目录"))

        self.history_path_button = QPushButton("选择")
        self.history_path_button.setObjectName("SecondaryButton")
        self.history_path_button.clicked.connect(self._choose_history_file)

        form.addRow(self._section_label("输出路径"))
        self._add_form_row(
            form,
            "结果目录",
            self._path_row(self.result_dir_edit, self.result_dir_button),
            "单张检测JSON、批量检测CSV和记录导出CSV的默认保存目录。",
        )
        self._add_form_row(
            form,
            "热力图目录",
            self._path_row(self.heatmap_dir_edit, self.heatmap_dir_button),
            "Grad-CAM原图、热力图和叠加图的默认保存目录。",
        )
        self._add_form_row(
            form,
            "记录文件",
            self._path_row(self.history_path_edit, self.history_path_button),
            "记录管理和数据统计读取的JSON文件。",
        )
        return form

    def _build_action_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("ResultBox")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.config_path_label = QLabel(f"配置文件：{display_path(CONFIG_PATH)}")
        self.config_path_label.setObjectName("MutedText")
        self.config_path_label.setWordWrap(True)

        self.status_label = QLabel("修改后点击保存设置。")
        self.status_label.setObjectName("MutedText")
        self.status_label.setWordWrap(True)

        button_row = QHBoxLayout()
        self.save_button = QPushButton("保存设置")
        self.save_button.setObjectName("PrimaryButton")
        self.save_button.clicked.connect(self.save_settings)

        self.reset_button = QPushButton("恢复默认")
        self.reset_button.setObjectName("SecondaryButton")
        self.reset_button.clicked.connect(self.reset_defaults)

        button_row.addWidget(self.save_button)
        button_row.addWidget(self.reset_button)
        button_row.addStretch()

        layout.addWidget(self.config_path_label)
        layout.addWidget(self.status_label)
        layout.addLayout(button_row)
        return frame

    def load_settings(self, config: dict):
        self.refresh_model_options(config.get("default_model", "aide"))
        self._set_combo_data(self.default_model_combo, config.get("default_model", "aide"))
        self._set_combo_data(self.device_combo, config.get("device", "cuda"))

        paths = config.get("paths", {})
        self.result_dir_edit.setText(str(paths.get("results", "output/results")))
        self.heatmap_dir_edit.setText(str(paths.get("heatmaps", "output/heatmaps")))
        self.history_path_edit.setText(str(paths.get("history", "data/history.json")))

        behavior = config.get("behavior", {})
        self.save_history_checkbox.setChecked(bool(behavior.get("save_history", True)))
        self.auto_gradcam_checkbox.setChecked(bool(behavior.get("auto_gradcam", True)))
        self.status_label.setText("设置已从配置文件加载。")

    def refresh_model_options(self, selected_key: str | None = None):
        current_key = selected_key
        if hasattr(self, "default_model_combo") and current_key is None:
            current_key = self.default_model_combo.currentData()

        combo = getattr(self, "default_model_combo", None)
        if combo is None:
            return

        was_blocked = combo.blockSignals(True)
        combo.clear()
        for key, spec in MODEL_SPECS.items():
            combo.addItem(spec["name"], key)
        if current_key:
            self._set_combo_data(combo, current_key)
        combo.blockSignals(was_blocked)

    def set_current_model(self, model_key: str):
        self.refresh_model_options(model_key)
        self._set_combo_data(self.default_model_combo, model_key)

    def set_current_device(self, device: str):
        self._set_combo_data(self.device_combo, device)

    def save_settings(self):
        config = load_config()
        selected_model = self.default_model_combo.currentData()
        selected_device = self.device_combo.currentData()
        if selected_model not in MODEL_SPECS:
            QMessageBox.warning(self, "保存设置", "默认模型不存在，请重新选择。")
            return

        config["default_model"] = selected_model
        config["device"] = selected_device or "cuda"
        config.setdefault("paths", {})
        config["paths"]["results"] = self.result_dir_edit.text().strip() or "output/results"
        config["paths"]["heatmaps"] = self.heatmap_dir_edit.text().strip() or "output/heatmaps"
        config["paths"]["history"] = self.history_path_edit.text().strip() or "data/history.json"
        config["behavior"] = {
            "save_history": self.save_history_checkbox.isChecked(),
            "auto_gradcam": self.auto_gradcam_checkbox.isChecked(),
        }

        if not self._ensure_paths(config):
            return

        saved_config = save_config(config)
        self.status_label.setText("设置已保存，并已通知主窗口同步。")
        self.settings_saved.emit(saved_config)

    def reset_defaults(self):
        self.load_settings(default_config())
        self.status_label.setText("已恢复默认值预览，点击保存设置后生效。")

    def _ensure_paths(self, config: dict) -> bool:
        for key, title in (("results", "结果目录"), ("heatmaps", "热力图目录")):
            directory = path_for(key, config)
            if directory.exists() and not directory.is_dir():
                QMessageBox.warning(self, "保存设置", f"{title}不是文件夹：{directory}")
                return False
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                QMessageBox.warning(self, "保存设置", f"{title}无法创建：{directory}\n{exc}")
                return False

        history_path = path_for("history", config)
        if history_path.exists() and history_path.is_dir():
            QMessageBox.warning(self, "保存设置", f"记录文件不能是文件夹：{history_path}")
            return False
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "保存设置", f"记录文件目录无法创建：{history_path.parent}\n{exc}")
            return False
        return True

    def _choose_directory(self, target: QLineEdit, title: str):
        current = resolve_path(target.text().strip() or ".")
        folder = QFileDialog.getExistingDirectory(self, title, str(current))
        if folder:
            target.setText(display_path(Path(folder)))

    def _choose_history_file(self):
        current = resolve_path(self.history_path_edit.text().strip() or "data/history.json")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "选择记录JSON文件",
            str(current),
            "JSON Files (*.json);;All Files (*)",
        )
        if file_path:
            self.history_path_edit.setText(display_path(Path(file_path)))

    def _new_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(14)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        return form

    def _path_row(self, edit: QLineEdit, button: QPushButton) -> QWidget:
        wrapper = QWidget()
        wrapper.setObjectName("SettingsPathField")
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        edit.setMinimumWidth(260)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return wrapper

    def _prompt_label(self, text: str, tooltip: str = "") -> QLabel:
        label = QLabel(text)
        label.setObjectName("SettingsPromptLabel")
        label.setMinimumWidth(96)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if tooltip:
            label.setToolTip(tooltip)
        return label

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("SettingsSectionLabel")
        return label

    def _add_form_row(self, form: QFormLayout, label_text: str, field, tooltip: str = ""):
        form.addRow(self._prompt_label(label_text, tooltip), field)
        if tooltip and hasattr(field, "setToolTip"):
            field.setToolTip(tooltip)

    def _set_combo_data(self, combo: QComboBox, data):
        index = combo.findData(data)
        if index >= 0:
            combo.setCurrentIndex(index)
