import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from models.model_registry import MODEL_SPECS
from ui.pages.common import page_header, page_root, panel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "Night-AIGC-Dataset"
DEFAULT_TRAINING_ROOT = PROJECT_ROOT / "output" / "training"
CUSTOM_WEIGHT_DIR = PROJECT_ROOT / "models" / "weights" / "custom_uploads"
CNN_TRAIN_SCRIPT = PROJECT_ROOT / "training" / "train_cnn.py"
AIDE_TRAIN_SCRIPT = PROJECT_ROOT / "training" / "train_aide_night.py"
PROJECT_PYTHON = PROJECT_ROOT / "runtime" / "python" / "python.exe"
DEFAULT_PYTHON = PROJECT_PYTHON if PROJECT_PYTHON.exists() else Path(sys.executable)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}
DATASET_SPLITS = ("train", "val", "test")
DATASET_CLASSES = ("real", "fake")


def resolve_project_path(value: str) -> Path:
    path = Path(str(value or "").strip()).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_project_path(path: Path) -> str:
    try:
        return Path(path).resolve(strict=False).relative_to(PROJECT_ROOT.resolve(strict=False)).as_posix()
    except Exception:
        return str(path)


def display_command_program(program: str) -> str:
    path = Path(program)
    if not path.is_absolute():
        return str(program)
    project_path = display_project_path(path)
    if project_path != str(path):
        return project_path
    return path.name


class TrainingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.process: QProcess | None = None
        self.log_file = None
        self.current_output_dir: Path | None = None

        layout = page_root(self)
        layout.addWidget(page_header("自定义模型训练", "选择已有模型或上传兼容权重，在构建好的数据集上继续训练并另存权重。"))

        workspace = QSplitter(Qt.Orientation.Horizontal)
        workspace.setChildrenCollapsible(False)
        workspace.setHandleWidth(8)

        config_panel, config_layout = panel("训练配置")
        config_panel.setMinimumWidth(430)
        config_panel.setMaximumWidth(820)

        config_content = QWidget()
        config_content.setObjectName("TrainingConfigContent")
        form = QFormLayout(config_content)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        self.model_combo = QComboBox()
        self.model_combo.addItem("AIDE-Night", "aide")
        self.model_combo.addItem("ResNet50", "resnet50")
        self.model_combo.addItem("EfficientNet-B0", "efficientnet_b0")
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)

        self.python_edit = QLineEdit()
        self.python_edit.setMinimumWidth(0)
        self.python_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        python_button = QPushButton("浏览")
        python_button.setObjectName("SecondaryButton")
        python_button.setFixedWidth(70)
        python_button.clicked.connect(self.choose_python)
        python_row = QHBoxLayout()
        python_row.addWidget(self.python_edit, 1)
        python_row.addWidget(python_button)

        self.dataset_root_edit = QLineEdit(display_project_path(DEFAULT_DATASET_ROOT))
        self.dataset_root_edit.setMinimumWidth(0)
        self.dataset_root_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        dataset_button = QPushButton("浏览")
        dataset_button.setObjectName("SecondaryButton")
        dataset_button.setFixedWidth(70)
        dataset_button.clicked.connect(self.choose_dataset_root)
        create_dataset_button = QPushButton("创建结构")
        create_dataset_button.setObjectName("SecondaryButton")
        create_dataset_button.setFixedWidth(88)
        create_dataset_button.clicked.connect(self.create_dataset_structure)
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(self.dataset_root_edit, 1)
        dataset_row.addWidget(dataset_button)
        dataset_row.addWidget(create_dataset_button)
        self.dataset_status_label = QLabel()
        self.dataset_status_label.setObjectName("TrainingDatasetStatus")
        self.dataset_status_label.setWordWrap(True)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setMinimumWidth(0)
        self.output_dir_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        output_button = QPushButton("浏览")
        output_button.setObjectName("SecondaryButton")
        output_button.setFixedWidth(70)
        output_button.clicked.connect(self.choose_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit, 1)
        output_row.addWidget(output_button)

        self.weight_source_combo = QComboBox()
        self.weight_source_combo.addItem("从当前系统权重再训练", "current")
        self.weight_source_combo.addItem("上传自定义兼容权重再训练", "uploaded")
        self.weight_source_combo.addItem("使用项目默认权重初始化", "default")
        self.weight_source_combo.currentIndexChanged.connect(self._refresh_command_preview)

        self.custom_weight_edit = QLineEdit()
        self.custom_weight_edit.setMinimumWidth(0)
        self.custom_weight_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.custom_weight_edit.setPlaceholderText("可上传 .pth 或 .pt 权重文件")
        choose_weight_button = QPushButton("选择")
        choose_weight_button.setObjectName("SecondaryButton")
        choose_weight_button.setFixedWidth(64)
        choose_weight_button.clicked.connect(self.choose_weight)
        upload_weight_button = QPushButton("上传到项目")
        upload_weight_button.setObjectName("SecondaryButton")
        upload_weight_button.setFixedWidth(92)
        upload_weight_button.clicked.connect(self.upload_weight)
        weight_row = QHBoxLayout()
        weight_row.addWidget(self.custom_weight_edit, 1)
        weight_row.addWidget(choose_weight_button)
        weight_row.addWidget(upload_weight_button)

        self.device_combo = QComboBox()
        self.device_combo.addItem("GPU (CUDA)", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.currentIndexChanged.connect(self._refresh_command_preview)

        self.class_weight_combo = QComboBox()
        self.class_weight_combo.addItem("balanced 自动平衡", "balanced")
        self.class_weight_combo.addItem("none 不加权", "none")
        self.class_weight_combo.currentIndexChanged.connect(self._refresh_command_preview)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(5)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(4)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(20260719)

        self.freeze_check = QCheckBox("ResNet/EfficientNet 快速训练：冻结骨干网络")
        self.freeze_check.setChecked(True)
        self.pretrained_check = QCheckBox("ResNet/EfficientNet 使用 ImageNet 预训练")
        self.pretrained_check.setChecked(True)
        self.auto_resume_check = QCheckBox("自动断点续训")
        self.auto_resume_check.setChecked(False)
        self.test_after_train_check = QCheckBox("训练完成后自动测试")
        self.test_after_train_check.setChecked(True)
        self.amp_check = QCheckBox("AIDE 使用 AMP 混合精度")
        self.amp_check.setChecked(True)
        self.hide_warnings_check = QCheckBox("隐藏普通 Python 警告")
        self.hide_warnings_check.setChecked(True)

        self.aide_scope_combo = QComboBox()
        self.aide_scope_combo.addItem("recommended 推荐训练范围", "recommended")
        self.aide_scope_combo.addItem("head 只训练头部", "head")
        self.aide_scope_combo.addItem("all_except_convnext", "all_except_convnext")

        self.accum_spin = QSpinBox()
        self.accum_spin.setRange(1, 128)
        self.accum_spin.setValue(8)

        form.addRow(self._section_label("基础配置"))
        self._add_form_row(form, "训练模型", self.model_combo, "选择要训练或继续训练的模型结构。")
        self._add_form_row(form, "Python解释器", python_row, "选择当前模型训练所使用的 Python 环境。")
        self._add_form_row(form, "数据集根目录", dataset_row, "选择包含 train、val、test 子目录的数据集根目录。")
        self._add_form_row(form, "输出目录", output_row, "训练日志、命令记录、checkpoint 和最终权重会保存到这里。")
        self._add_form_row(form, "权重来源", self.weight_source_combo, "选择从系统现有权重、自定义权重或项目默认权重开始训练。")
        self._add_form_row(form, "自定义权重", weight_row, "仅在权重来源选择上传自定义权重时使用。")
        self._add_form_row(form, "运行设备", self.device_combo, "选择使用 GPU CUDA 或 CPU 训练。")
        self._add_form_row(form, "Loss权重", self.class_weight_combo, "balanced 会按类别数量自动调整 loss 权重，适合类别不均衡数据集。")
        self._add_form_row(form, "训练轮数", self.epochs_spin, "epochs，每轮会遍历一次训练集。")
        self._add_form_row(form, "Batch大小", self.batch_spin, "batch size，显存不足时调小。")
        self._add_form_row(form, "读取线程", self.workers_spin, "num workers，用于并行读取图片；0 表示主进程读取。")
        self._add_form_row(form, "随机种子", self.seed_spin, "seed，用于控制数据打乱和随机增强，便于复现实验。")

        form.addRow(self._section_label("通用训练选项"))
        self._add_form_row(form, "断点续训", self.auto_resume_check, "自动从输出目录中最近的 checkpoint 继续训练。")
        self._add_form_row(form, "训练后测试", self.test_after_train_check, "训练结束后自动在 test 集上评估。")
        self._add_form_row(form, "警告输出", self.hide_warnings_check, "隐藏常见 Python 警告，保留训练进度、loss 和 acc 输出。")

        form.addRow(self._section_label("ResNet / EfficientNet 选项"))
        self._add_form_row(form, "冻结骨干", self.freeze_check, "只训练分类头，训练更快；取消勾选会训练更多参数。")
        self._add_form_row(form, "ImageNet预训练", self.pretrained_check, "使用 torchvision 的 ImageNet 初始权重。")

        form.addRow(self._section_label("AIDE 选项"))
        self._add_form_row(form, "AMP混合精度", self.amp_check, "使用混合精度训练以降低显存占用并加快训练。")
        self._add_form_row(form, "训练范围", self.aide_scope_combo, "recommended 是当前推荐的 AIDE 微调范围。")
        self._add_form_row(form, "累积步数", self.accum_spin, "accum steps，多个小 batch 累积后再更新一次参数。")

        for widget in [
            self.epochs_spin,
            self.batch_spin,
            self.workers_spin,
            self.seed_spin,
            self.freeze_check,
            self.pretrained_check,
            self.auto_resume_check,
            self.test_after_train_check,
            self.amp_check,
            self.aide_scope_combo,
            self.accum_spin,
            self.custom_weight_edit,
            self.dataset_root_edit,
            self.output_dir_edit,
        ]:
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._refresh_command_preview)
            if hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(self._refresh_command_preview)
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._refresh_command_preview)
            if hasattr(widget, "textChanged"):
                widget.textChanged.connect(self._refresh_command_preview)

        config_scroll = QScrollArea()
        config_scroll.setObjectName("TrainingConfigScroll")
        config_scroll.setWidgetResizable(True)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        config_scroll.setWidget(config_content)
        config_layout.addWidget(config_scroll, 1)
        config_layout.addWidget(self.dataset_status_label)

        action_row = QHBoxLayout()
        self.start_button = QPushButton("开始训练")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("停止训练")
        self.stop_button.setObjectName("SecondaryButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.stop_button)
        action_row.addStretch()
        config_layout.addLayout(action_row)

        preview_panel, preview_layout = panel("命令预览与训练输出")
        self.command_preview = QTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMinimumHeight(80)
        self.command_preview.setMaximumHeight(180)
        self.command_preview.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.training_status_label = QLabel("训练尚未启动。")
        self.training_status_label.setObjectName("TrainingRunStatus")
        self.training_status_label.setWordWrap(True)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(220)
        self.log_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        preview_layout.addWidget(QLabel("即将执行的命令"))
        preview_layout.addWidget(self.command_preview)
        preview_layout.addWidget(self.training_status_label)
        preview_layout.addWidget(self.progress)
        preview_layout.addWidget(self.log_edit, 1)

        workspace.addWidget(config_panel)
        workspace.addWidget(preview_panel)
        workspace.setStretchFactor(0, 0)
        workspace.setStretchFactor(1, 1)
        workspace.setSizes([560, 720])
        layout.addWidget(workspace, 1)

        self._on_model_changed()

    def _prompt_label(self, text: str, tooltip: str = "") -> QLabel:
        label = QLabel(text)
        label.setObjectName("TrainingPromptLabel")
        label.setMinimumWidth(112)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if tooltip:
            label.setToolTip(tooltip)
        return label

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("TrainingSectionLabel")
        return label

    def _add_form_row(self, form: QFormLayout, label_text: str, field, tooltip: str = ""):
        form.addRow(self._prompt_label(label_text, tooltip), field)
        if tooltip:
            self._apply_tooltip(field, tooltip)

    def _apply_tooltip(self, field, tooltip: str):
        if hasattr(field, "setToolTip"):
            field.setToolTip(tooltip)
            return
        if hasattr(field, "count"):
            for index in range(field.count()):
                item = field.itemAt(index)
                widget = item.widget() if item is not None else None
                if widget is not None:
                    widget.setToolTip(tooltip)

    def choose_python(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择Python解释器", self.python_edit.text(), "Python (*.exe);;All Files (*)")
        if file_path:
            self.python_edit.setText(display_project_path(Path(file_path)))

    def choose_dataset_root(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据集根目录", str(resolve_project_path(self.dataset_root_edit.text())))
        if folder:
            self.dataset_root_edit.setText(display_project_path(Path(folder)))

    def choose_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择训练输出目录", str(resolve_project_path(self.output_dir_edit.text())))
        if folder:
            self.output_dir_edit.setText(display_project_path(Path(folder)))

    def create_dataset_structure(self):
        dataset_text = self.dataset_root_edit.text().strip()
        if not dataset_text:
            QMessageBox.warning(self, "数据集目录", "请先设置数据集根目录。")
            return
        dataset_root = resolve_project_path(dataset_text)
        self._ensure_dataset_structure(dataset_root)
        self._refresh_dataset_status(dataset_root)
        self._refresh_command_preview()
        QMessageBox.information(
            self,
            "数据集目录",
            (
                "已创建数据集目录结构。\n\n"
                f"{display_project_path(dataset_root)}\n\n"
                "请进入“数据集构建”页面导入 real 和 fake 两类图片，"
                "或手动放入 train/val/test 下的 real、fake 文件夹后再开始训练。"
            ),
        )

    def _ensure_dataset_structure(self, dataset_root: Path):
        for split in DATASET_SPLITS:
            for class_name in DATASET_CLASSES:
                (dataset_root / split / class_name).mkdir(parents=True, exist_ok=True)
                (dataset_root / "labels" / split / class_name).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels").mkdir(parents=True, exist_ok=True)

    def _dataset_counts(self, dataset_root: Path) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for split in DATASET_SPLITS:
            counts[split] = {}
            for class_name in DATASET_CLASSES:
                class_dir = dataset_root / split / class_name
                alt_dir = dataset_root / split / ("0_real" if class_name == "real" else "1_fake")
                actual_dir = class_dir if class_dir.exists() else alt_dir
                if not actual_dir.exists():
                    counts[split][class_name] = -1
                    continue
                counts[split][class_name] = sum(
                    1
                    for path in actual_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
        return counts

    def _refresh_dataset_status(self, dataset_root: Path | None = None) -> dict[str, dict[str, int]]:
        dataset_root = dataset_root or resolve_project_path(self.dataset_root_edit.text())
        counts = self._dataset_counts(dataset_root) if dataset_root.exists() else {}
        if not dataset_root.exists():
            self.dataset_status_label.setText(
                f"数据集尚未建立：{display_project_path(dataset_root)}。请先到“数据集构建”页面导入图片，"
                "或点击“创建结构”建立 train/val/test/real/fake 目录。"
            )
            return counts

        parts = []
        missing = []
        empty = []
        for split in DATASET_SPLITS:
            split_counts = counts.get(split, {})
            for class_name in DATASET_CLASSES:
                value = split_counts.get(class_name, -1)
                if value < 0:
                    missing.append(f"{split}/{class_name}")
                elif value == 0:
                    empty.append(f"{split}/{class_name}")
            if split_counts:
                parts.append(
                    f"{split}: real={max(split_counts.get('real', 0), 0)}, "
                    f"fake={max(split_counts.get('fake', 0), 0)}"
                )

        if missing:
            self.dataset_status_label.setText(
                f"数据集结构不完整：缺少 {', '.join(missing)}。请先建立数据集目录结构。"
            )
        elif empty:
            self.dataset_status_label.setText(
                f"数据集目录已存在，但仍有空类别：{', '.join(empty)}。"
                "请先通过“数据集构建”页面导入 real 和 fake 图片。"
            )
        else:
            self.dataset_status_label.setText("数据集可用于训练；" + " | ".join(parts))
        return counts

    def _validate_dataset_for_training(self, dataset_root: Path, model_key: str):
        if not dataset_root.exists():
            raise ValueError(
                f"数据集尚未建立：{display_project_path(dataset_root)}\n"
                "请先进入“数据集构建”页面导入 real/fake 图片，"
                "或点击“创建结构”建立目录后再补充图片。"
            )
        counts = self._dataset_counts(dataset_root)
        required_splits = ["train", "val"]
        if self.test_after_train_check.isChecked():
            required_splits.append("test")
        missing = []
        empty = []
        for split in required_splits:
            for class_name in DATASET_CLASSES:
                value = counts.get(split, {}).get(class_name, -1)
                if value < 0:
                    missing.append(f"{split}/{class_name}")
                elif value == 0:
                    empty.append(f"{split}/{class_name}")
        if missing:
            raise ValueError(
                "数据集结构不完整：缺少 "
                + ", ".join(missing)
                + "。请先建立 train/val/test 下的 real、fake 目录。"
            )
        if empty:
            raise ValueError(
                "数据集图片不足："
                + ", ".join(empty)
                + " 为空。请先通过“数据集构建”页面导入真实图像和 AIGC 图像。"
            )

    def choose_weight(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择权重文件", str(CUSTOM_WEIGHT_DIR), "Weights (*.pth *.pt);;All Files (*)")
        if file_path:
            self.custom_weight_edit.setText(display_project_path(Path(file_path)))
            self.weight_source_combo.setCurrentIndex(self.weight_source_combo.findData("uploaded"))

    def upload_weight(self):
        source = resolve_project_path(self.custom_weight_edit.text())
        if not source.exists() or not source.is_file():
            QMessageBox.warning(self, "上传权重", "请先选择存在的权重文件。")
            return
        CUSTOM_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = CUSTOM_WEIGHT_DIR / f"{source.stem}_{timestamp}{source.suffix}"
        shutil.copy2(source, target)
        self.custom_weight_edit.setText(display_project_path(target))
        self.weight_source_combo.setCurrentIndex(self.weight_source_combo.findData("uploaded"))
        QMessageBox.information(self, "上传权重", f"权重已复制到项目：\n{target}")

    def start_training(self):
        if self.process is not None:
            QMessageBox.information(self, "模型训练", "当前已有训练任务正在运行。")
            return

        try:
            program, args, output_dir = self._build_command()
        except ValueError as exc:
            self.training_status_label.setText(str(exc))
            self._refresh_dataset_status()
            QMessageBox.warning(self, "训练配置无效", str(exc))
            return

        reply = QMessageBox.question(
            self,
            "开始训练",
            f"训练会占用较长时间并写入输出目录：\n{output_dir}\n\n是否开始？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        command_path = output_dir / "train_command.txt"
        command_path.write_text(subprocess.list2cmdline([program, *args]), encoding="utf-8")
        self.log_file = (output_dir / "train_stdout.log").open("a", encoding="utf-8")
        self.current_output_dir = output_dir

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        if self.hide_warnings_check.isChecked():
            env.insert("PYTHONWARNINGS", "ignore")
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._on_process_finished)
        self.process.errorOccurred.connect(self._on_process_error)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.training_status_label.setText(f"训练已启动，输出目录：{display_project_path(output_dir)}")
        self.log_edit.clear()
        self._append_log(f"启动训练：{subprocess.list2cmdline([display_command_program(program), *args])}")
        self._append_log(f"输出目录：{display_project_path(output_dir)}")
        self.process.start(program, args)

    def stop_training(self):
        if self.process is None:
            return
        reply = QMessageBox.warning(
            self,
            "停止训练",
            "是否停止当前训练进程？已保存的 checkpoint 不会被删除。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.process.terminate()

    def _build_command(self, validate_dataset: bool = True) -> tuple[str, list[str], Path]:
        model_key = str(self.model_combo.currentData())
        python_text = self.python_edit.text().strip()
        dataset_text = self.dataset_root_edit.text().strip()
        output_text = self.output_dir_edit.text().strip()
        if not python_text:
            raise ValueError("请设置Python解释器。")
        if not dataset_text:
            raise ValueError("请设置数据集根目录。")
        if not output_text:
            raise ValueError("请设置训练输出目录。")
        python_path = resolve_project_path(python_text)
        dataset_root = resolve_project_path(dataset_text)
        output_dir = resolve_project_path(output_text)
        if not python_path.exists():
            raise ValueError(f"Python解释器不存在：{python_path}")
        if validate_dataset:
            self._validate_dataset_for_training(dataset_root, model_key)

        if model_key == "aide":
            script_path = AIDE_TRAIN_SCRIPT
            if not script_path.exists():
                raise ValueError(f"AIDE训练脚本不存在：{script_path}")
            args = [
                "-u",
                display_project_path(script_path),
                "--mode",
                "train",
                "--dataset_root",
                display_project_path(dataset_root),
                "--output_dir",
                display_project_path(output_dir),
                "--epochs",
                str(self.epochs_spin.value()),
                "--batch_size",
                str(self.batch_spin.value()),
                "--accum_steps",
                str(self.accum_spin.value()),
                "--num_workers",
                str(self.workers_spin.value()),
                "--device",
                str(self.device_combo.currentData()),
                "--seed",
                str(self.seed_spin.value()),
                "--train_scope",
                str(self.aide_scope_combo.currentData()),
                "--class_weight",
                str(self.class_weight_combo.currentData()),
            ]
            pretrained = self._selected_checkpoint_path()
            if pretrained is not None:
                args.extend(["--pretrained_checkpoint", display_project_path(pretrained)])
            if self.amp_check.isChecked():
                args.append("--amp")
            if self.auto_resume_check.isChecked():
                args.append("--auto_resume")
            if self.test_after_train_check.isChecked():
                args.append("--test_after_train")
            return str(python_path), args, output_dir

        if model_key not in {"resnet50", "efficientnet_b0"}:
            raise ValueError(f"暂不支持的模型：{model_key}")

        script_path = CNN_TRAIN_SCRIPT
        if not script_path.exists():
            raise ValueError(f"ResNet/EfficientNet训练脚本不存在：{script_path}")
        train_dir = dataset_root / "train"
        val_dir = dataset_root / "val"
        test_dir = dataset_root / "test"
        if validate_dataset and not train_dir.exists():
            raise ValueError(f"训练集目录不存在：{train_dir}")
        args = [
            "-u",
            display_project_path(script_path),
            "--mode",
            "train",
            "--model",
            model_key,
            "--train_dir",
            display_project_path(train_dir),
            "--val_dir",
            display_project_path(val_dir),
            "--test_dir",
            display_project_path(test_dir),
            "--result_dir",
            display_project_path(output_dir),
            "--epochs",
            str(self.epochs_spin.value()),
            "--batch_size",
            str(self.batch_spin.value()),
            "--num_workers",
            str(self.workers_spin.value()),
            "--device",
            str(self.device_combo.currentData()),
            "--seed",
            str(self.seed_spin.value()),
            "--class_weight",
            str(self.class_weight_combo.currentData()),
        ]
        if self.pretrained_check.isChecked():
            args.append("--pretrained")
        if self.freeze_check.isChecked():
            args.append("--freeze_backbone")
        resume_path = self._selected_checkpoint_path()
        if resume_path is not None:
            args.extend(["--resume_checkpoint", display_project_path(resume_path)])
        if self.auto_resume_check.isChecked():
            args.append("--auto_resume")
        if self.test_after_train_check.isChecked():
            args.append("--test_after_train")
        return str(python_path), args, output_dir

    def _selected_checkpoint_path(self) -> Path | None:
        source = self.weight_source_combo.currentData()
        if source == "default":
            return None
        if source == "uploaded":
            path = resolve_project_path(self.custom_weight_edit.text())
            if not path.exists():
                raise ValueError("已选择上传自定义权重，但权重路径不存在。")
            return path

        model_key = str(self.model_combo.currentData())
        spec = MODEL_SPECS[model_key]
        path = Path(spec["weight_path"])
        if not path.exists():
            raise ValueError(f"当前系统权重不存在：{path}")
        return path

    def _read_process_output(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput())
        text = self._decode_output(data)
        if text:
            self._append_log(text.rstrip())

    def _on_process_finished(self, exit_code: int, exit_status):
        self._read_process_output()
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if exit_code == 0 else 0)
        status = "完成" if exit_code == 0 else f"结束，退出码 {exit_code}"
        output_text = display_project_path(self.current_output_dir) if self.current_output_dir else ""
        self.training_status_label.setText(f"训练进程{status}。输出目录：{output_text}")
        self._append_log(f"训练进程{status}。输出目录：{output_text}")
        self._close_log_file()
        self.process = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _on_process_error(self, error):
        self.training_status_label.setText(f"训练进程错误：{error}")
        self._append_log(f"训练进程错误：{error}")

    def _append_log(self, text: str):
        self.log_edit.append(text)
        if self.log_file is not None:
            self.log_file.write(text + "\n")
            self.log_file.flush()

    def _close_log_file(self):
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def _decode_output(self, data: bytes) -> str:
        for encoding in ("utf-8", "gbk", "mbcs"):
            try:
                return data.decode(encoding)
            except Exception:
                continue
        return data.decode(errors="replace")

    def _on_model_changed(self):
        model_key = str(self.model_combo.currentData())
        self.python_edit.setText(display_project_path(DEFAULT_PYTHON))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir_edit.setText(display_project_path(DEFAULT_TRAINING_ROOT / f"{model_key}_{timestamp}"))
        is_aide = model_key == "aide"
        self.amp_check.setEnabled(is_aide)
        self.aide_scope_combo.setEnabled(is_aide)
        self.accum_spin.setEnabled(is_aide)
        self.freeze_check.setEnabled(not is_aide)
        self.pretrained_check.setEnabled(not is_aide)
        self.batch_spin.setValue(2 if is_aide else 16)
        self.epochs_spin.setValue(3 if is_aide else 5)
        self._refresh_command_preview()

    def _refresh_command_preview(self, *args):
        dataset_root = resolve_project_path(self.dataset_root_edit.text())
        counts = self._refresh_dataset_status(dataset_root)
        try:
            program, command_args, _ = self._build_command(validate_dataset=False)
            command_text = subprocess.list2cmdline([display_command_program(program), *command_args])
            if not dataset_root.exists():
                text = (
                    f"数据集尚未建立：{display_project_path(dataset_root)}\n"
                    "请先进入“数据集构建”页面导入 real/fake 图片，"
                    "或点击左侧“创建结构”建立目录后再补充图片。\n\n"
                    "数据集建立完成后将执行：\n"
                    f"{command_text}"
                )
                self.training_status_label.setText("等待建立数据集，当前不会启动训练。")
            else:
                empty = [
                    f"{split}/{class_name}"
                    for split in DATASET_SPLITS
                    for class_name in DATASET_CLASSES
                    if counts.get(split, {}).get(class_name, -1) == 0
                ]
                text = command_text
                if empty:
                    text = (
                        "数据集目录已存在，但部分类别仍为空："
                        + ", ".join(empty)
                        + "\n请先通过“数据集构建”页面导入图片。\n\n"
                        + "图片补齐后将执行：\n"
                        + command_text
                    )
                    self.training_status_label.setText("等待补齐数据集图片，当前不会启动训练。")
                else:
                    self.training_status_label.setText("配置已可用于训练。开始训练前仍会进行数据集完整性检查。")
        except Exception as exc:
            text = f"配置尚未完整：{exc}"
            self.training_status_label.setText("请先补全训练配置。")
        self.command_preview.setPlainText(text)
