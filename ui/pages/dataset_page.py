import csv
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
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

from ui.pages.common import page_header, page_root, panel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "Night-AIGC-Dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}
SPLITS = ("train", "val", "test")
CLASS_LABELS = {"real": 0, "fake": 1}
CSV_HEADER = ["split", "class", "label", "image", "source_path", "imported_at"]


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


@dataclass
class DatasetBuildSettings:
    source_dir: Path
    dataset_root: Path
    class_name: str
    split_mode: str
    ratios: dict[str, float]
    prefix: str
    digits: int
    seed: int
    shuffle: bool
    allow_duplicates: bool
    dry_run: bool


class DatasetBuildWorker(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, settings: DatasetBuildSettings):
        super().__init__()
        self.settings = settings

    def run(self):
        try:
            plan, skipped = build_import_plan(self.settings)
            if self.settings.dry_run:
                counts = Counter(item["split"] for item in plan)
                self.finished.emit(
                    {
                        "dry_run": True,
                        "planned": len(plan),
                        "skipped": len(skipped),
                        "counts": dict(counts),
                    }
                )
                return

            ensure_dataset_dirs(self.settings.dataset_root, self.settings.class_name)
            imported_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rows_by_split = {split: [] for split in SPLITS}
            label_entries_by_split = {split: [] for split in SPLITS}
            success = 0
            total = len(plan)

            for item in plan:
                shutil.copy2(item["source_path"], item["dest_path"])
                row = {
                    "split": item["split"],
                    "class": self.settings.class_name,
                    "label": CLASS_LABELS[self.settings.class_name],
                    "image": item["filename"],
                    "source_path": str(item["source_path"]),
                    "imported_at": imported_at,
                }
                rows_by_split[item["split"]].append(row)
                label_entries_by_split[item["split"]].append((item["filename"], row["label"]))
                success += 1
                self.progress.emit(success, total)

            all_rows = []
            for split in SPLITS:
                rows = rows_by_split[split]
                if not rows:
                    continue
                append_csv_rows(self.settings.dataset_root / "labels" / f"{split}_labels.csv", rows)
                append_label_txt(
                    self.settings.dataset_root / "labels" / split / self.settings.class_name / "labels.txt",
                    label_entries_by_split[split],
                )
                all_rows.extend(rows)

            if all_rows:
                append_csv_rows(self.settings.dataset_root / "labels" / "all_labels.csv", all_rows)

            update_summary(self.settings.dataset_root)
            counts = Counter(item["split"] for item in plan)
            self.finished.emit(
                {
                    "dry_run": False,
                    "planned": len(plan),
                    "imported": success,
                    "skipped": len(skipped),
                    "counts": dict(counts),
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetPage(QWidget):
    def __init__(self):
        super().__init__()
        self.worker: DatasetBuildWorker | None = None
        self.thread: QThread | None = None

        layout = page_root(self)
        layout.addWidget(page_header("数据集构建", "导入真实或AIGC图片，自动分配 train/val/test，复制、命名并写入 labels。"))

        build_panel, build_layout = panel("导入图片到数据集")
        config_scroll = QScrollArea()
        config_scroll.setObjectName("DatasetConfigScroll")
        config_scroll.setWidgetResizable(True)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        config_scroll.setMinimumHeight(180)

        config_content = QWidget()
        config_content.setObjectName("DatasetConfigContent")
        form = QVBoxLayout(config_content)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("选择待导入图片文件夹")
        self.source_edit.setMinimumWidth(220)
        self.source_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.dataset_root_edit = QLineEdit(display_project_path(DEFAULT_DATASET_ROOT))
        self.dataset_root_edit.setMinimumWidth(220)
        self.dataset_root_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        source_button = QPushButton("浏览")
        source_button.setObjectName("SecondaryButton")
        source_button.setFixedWidth(76)
        source_button.clicked.connect(self.choose_source_dir)
        dataset_button = QPushButton("浏览")
        dataset_button.setObjectName("SecondaryButton")
        dataset_button.setFixedWidth(76)
        dataset_button.clicked.connect(self.choose_dataset_root)

        source_row = QHBoxLayout()
        source_row.setContentsMargins(0, 0, 0, 0)
        source_row.setSpacing(8)
        source_row.addWidget(self.source_edit, 1)
        source_row.addWidget(source_button)
        dataset_row = QHBoxLayout()
        dataset_row.setContentsMargins(0, 0, 0, 0)
        dataset_row.setSpacing(8)
        dataset_row.addWidget(self.dataset_root_edit, 1)
        dataset_row.addWidget(dataset_button)

        self.class_combo = QComboBox()
        self.class_combo.addItem("真实图像 real", "real")
        self.class_combo.addItem("AIGC图像 fake", "fake")

        self.split_combo = QComboBox()
        self.split_combo.addItem("自动分配 train/val/test", "auto")
        self.split_combo.addItem("全部导入 train", "train")
        self.split_combo.addItem("全部导入 val", "val")
        self.split_combo.addItem("全部导入 test", "test")

        self.ratio_edit = QLineEdit("0.70,0.15,0.15")
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("可选，例如 real 或 aigc")
        self.digits_spin = QSpinBox()
        self.digits_spin.setRange(3, 8)
        self.digits_spin.setValue(5)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(20260719)
        self.shuffle_check = QCheckBox("随机打乱后分配")
        self.shuffle_check.setChecked(True)
        self.duplicate_check = QCheckBox("允许重复导入同一来源图片")
        self.duplicate_check.setChecked(False)

        form.addWidget(self._section_label("来源与目标"))
        self._add_form_row(form, "源图片文件夹", source_row, "选择要导入的数据来源文件夹，工具会自动遍历常见图片格式。")
        self._add_form_row(form, "数据集根目录", dataset_row, "选择 Night-AIGC-Dataset 根目录，图片会复制到对应 split/class 目录。")

        form.addWidget(self._section_label("标注与分配"))
        self._add_form_row(form, "类别标注", self.class_combo, "real 表示真实图像，fake 表示 AIGC 图像。")
        self._add_form_row(form, "分配方式", self.split_combo, "选择自动分配 train/val/test，或全部导入某一个集合。")
        self._add_form_row(form, "分配比例", self.ratio_edit, "自动分配时使用，格式为 train,val,test，例如 0.70,0.15,0.15。")
        self._add_form_row(form, "随机打乱", self.shuffle_check, "勾选后先打乱图片顺序再按比例分配。")
        self._add_form_row(form, "随机种子", self.seed_spin, "seed 控制随机打乱结果，保持同一 seed 可复现分配。")

        form.addWidget(self._section_label("命名与去重"))
        self._add_form_row(form, "命名前缀", self.prefix_edit, "可选前缀，例如 real 或 aigc；为空时使用类别名称。")
        self._add_form_row(form, "编号位数", self.digits_spin, "自动命名编号位数，例如 5 表示 00001、00002。")
        self._add_form_row(form, "重复导入", self.duplicate_check, "默认跳过已经导入过的来源图片；勾选后允许重复导入。")
        form.addStretch()
        config_scroll.setWidget(config_content)

        action_row = QHBoxLayout()
        self.preview_button = QPushButton("预览导入计划")
        self.preview_button.setObjectName("SecondaryButton")
        self.preview_button.clicked.connect(lambda: self.start_build(dry_run=True))
        self.import_button = QPushButton("导入到数据集")
        self.import_button.setObjectName("PrimaryButton")
        self.import_button.clicked.connect(lambda: self.start_build(dry_run=False))
        action_row.addWidget(self.preview_button)
        action_row.addWidget(self.import_button)
        action_row.addStretch()

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(140)

        output_area = QWidget()
        output_layout = QVBoxLayout(output_area)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(10)
        output_layout.addLayout(action_row)
        output_layout.addWidget(self.progress)
        output_layout.addWidget(self.log_edit, 1)

        page_splitter = QSplitter(Qt.Orientation.Vertical)
        page_splitter.setChildrenCollapsible(False)
        page_splitter.setHandleWidth(8)
        page_splitter.addWidget(config_scroll)
        page_splitter.addWidget(output_area)
        page_splitter.setStretchFactor(0, 1)
        page_splitter.setStretchFactor(1, 1)
        page_splitter.setSizes([340, 240])
        build_layout.addWidget(page_splitter, 1)
        layout.addWidget(build_panel, 1)

    def _prompt_label(self, text: str, tooltip: str = "") -> QLabel:
        label = QLabel(text)
        label.setObjectName("DatasetPromptLabel")
        label.setMinimumWidth(104)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if tooltip:
            label.setToolTip(tooltip)
        return label

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("DatasetSectionLabel")
        return label

    def _add_form_row(self, form: QVBoxLayout, label_text: str, field, tooltip: str = ""):
        row = QWidget()
        row.setObjectName("DatasetFormRow")
        row.setMinimumHeight(42)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(12)

        row_layout.addWidget(self._prompt_label(label_text, tooltip))
        if isinstance(field, QHBoxLayout):
            field_widget = QWidget()
            field_widget.setObjectName("DatasetFieldBox")
            field_widget.setLayout(field)
            row_layout.addWidget(field_widget, 1)
        else:
            field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row_layout.addWidget(field, 1)

        form.addWidget(row)
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

    def choose_source_dir(self):
        start_dir = str(resolve_project_path(self.source_edit.text())) if self.source_edit.text().strip() else str(PROJECT_ROOT)
        folder = QFileDialog.getExistingDirectory(self, "选择源图片文件夹", start_dir)
        if folder:
            self.source_edit.setText(display_project_path(Path(folder)))

    def choose_dataset_root(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据集根目录", str(resolve_project_path(self.dataset_root_edit.text())))
        if folder:
            self.dataset_root_edit.setText(display_project_path(Path(folder)))

    def start_build(self, dry_run: bool):
        if self.thread is not None:
            QMessageBox.information(self, "数据集构建", "当前已有导入任务正在运行。")
            return

        settings = self._read_settings(dry_run)
        if settings is None:
            return

        if not dry_run:
            reply = QMessageBox.question(
                self,
                "导入到数据集",
                "将复制图片到数据集目录，并写入 labels CSV、labels.txt 和 summary.json。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.log_edit.clear()
        self.progress.setValue(0)
        self._set_controls_enabled(False)
        self.thread = QThread(self)
        self.worker = DatasetBuildWorker(settings)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._clear_worker)
        self.thread.start()

    def _read_settings(self, dry_run: bool) -> DatasetBuildSettings | None:
        source_text = self.source_edit.text().strip()
        dataset_text = self.dataset_root_edit.text().strip()
        if not source_text:
            QMessageBox.warning(self, "源目录无效", "请选择源图片文件夹。")
            return None
        if not dataset_text:
            QMessageBox.warning(self, "数据集目录无效", "请设置数据集根目录。")
            return None
        source_dir = resolve_project_path(source_text)
        dataset_root = resolve_project_path(dataset_text)
        if not source_dir.exists() or not source_dir.is_dir():
            QMessageBox.warning(self, "源目录无效", "请选择存在的源图片文件夹。")
            return None
        if source_dir.resolve() == dataset_root.resolve() or is_subpath(source_dir, dataset_root):
            QMessageBox.warning(self, "目录无效", "源目录不能位于数据集内部。")
            return None
        try:
            ratios = parse_ratios(self.ratio_edit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "比例无效", str(exc))
            return None
        prefix = sanitize_prefix(self.prefix_edit.text().strip())
        return DatasetBuildSettings(
            source_dir=source_dir,
            dataset_root=dataset_root,
            class_name=str(self.class_combo.currentData()),
            split_mode=str(self.split_combo.currentData()),
            ratios=ratios,
            prefix=prefix,
            digits=int(self.digits_spin.value()),
            seed=int(self.seed_spin.value()),
            shuffle=self.shuffle_check.isChecked(),
            allow_duplicates=self.duplicate_check.isChecked(),
            dry_run=dry_run,
        )

    def _update_progress(self, done: int, total: int):
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)

    def _on_finished(self, result: dict):
        self._set_controls_enabled(True)
        counts = result.get("counts", {})
        text = (
            f"预览完成：计划导入 {result.get('planned', 0)} 张，跳过 {result.get('skipped', 0)} 张。"
            if result.get("dry_run")
            else f"导入完成：成功 {result.get('imported', 0)} 张，跳过 {result.get('skipped', 0)} 张。"
        )
        self.log_edit.append(text)
        self.log_edit.append(f"train={counts.get('train', 0)} | val={counts.get('val', 0)} | test={counts.get('test', 0)}")
        self.log_edit.append(f"数据集目录：{self.dataset_root_edit.text()}")

    def _on_failed(self, message: str):
        self._set_controls_enabled(True)
        self.log_edit.append(f"任务失败：{message}")
        QMessageBox.warning(self, "数据集构建失败", message)

    def _clear_worker(self):
        self.thread = None
        self.worker = None

    def _set_controls_enabled(self, enabled: bool):
        self.preview_button.setEnabled(enabled)
        self.import_button.setEnabled(enabled)


def collect_images(folder: Path) -> list[Path]:
    return [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def parse_ratios(text: str) -> dict[str, float]:
    parts = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("比例必须是 train,val,test 三个数字，例如 0.70,0.15,0.15。")
    if any(part < 0 for part in parts):
        raise ValueError("比例不能为负数。")
    total = sum(parts)
    if total <= 0:
        raise ValueError("比例总和必须大于 0。")
    return {split: part / total for split, part in zip(SPLITS, parts)}


def split_images(images: list[Path], settings: DatasetBuildSettings) -> list[tuple[str, Path]]:
    image_list = list(images)
    if settings.shuffle:
        random.Random(settings.seed).shuffle(image_list)
    if settings.split_mode in SPLITS:
        return [(settings.split_mode, image) for image in image_list]

    total = len(image_list)
    train_count = round(total * settings.ratios["train"])
    val_count = round(total * settings.ratios["val"])
    if train_count + val_count > total:
        val_count = max(0, total - train_count)
    test_count = total - train_count - val_count
    split_sequence = ["train"] * train_count + ["val"] * val_count + ["test"] * test_count
    return list(zip(split_sequence, image_list))


def build_import_plan(settings: DatasetBuildSettings) -> tuple[list[dict], list[Path]]:
    images = collect_images(settings.source_dir)
    existing_sources = set() if settings.allow_duplicates else read_existing_sources(settings.dataset_root)
    skipped = [image for image in images if str(image.resolve()).lower() in existing_sources]
    images = [image for image in images if image not in skipped]

    next_indices = next_output_indices(settings.dataset_root, settings.class_name, settings.prefix)
    plan = []
    for split, source_path in split_images(images, settings):
        class_dir = settings.dataset_root / split / settings.class_name
        index = next_indices[split]
        while True:
            name_stem = f"{settings.prefix}_{index:0{settings.digits}d}" if settings.prefix else f"{index:0{settings.digits}d}"
            filename = f"{name_stem}{source_path.suffix.lower()}"
            dest_path = class_dir / filename
            if not dest_path.exists():
                break
            index += 1
        next_indices[split] = index + 1
        plan.append(
            {
                "split": split,
                "source_path": source_path,
                "dest_path": dest_path,
                "filename": filename,
            }
        )
    return plan, skipped


def next_output_indices(dataset_root: Path, class_name: str, prefix: str) -> dict[str, int]:
    result = {}
    if prefix:
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.[^.]+$", re.IGNORECASE)
    else:
        pattern = re.compile(r"^(\d+)\.[^.]+$", re.IGNORECASE)
    for split in SPLITS:
        max_index = 0
        class_dir = dataset_root / split / class_name
        if class_dir.exists():
            for item in class_dir.iterdir():
                if not item.is_file():
                    continue
                match = pattern.match(item.name)
                if match:
                    max_index = max(max_index, int(match.group(1)))
        result[split] = max_index + 1
    return result


def ensure_dataset_dirs(dataset_root: Path, class_name: str):
    for split in SPLITS:
        (dataset_root / split / class_name).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split / class_name).mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels").mkdir(parents=True, exist_ok=True)


def append_csv_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADER)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def append_label_txt(path: Path, entries: list[tuple[str, int]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for filename, label in entries:
            file.write(f"{filename} {label}\n")


def read_existing_sources(dataset_root: Path) -> set[str]:
    csv_path = dataset_root / "labels" / "all_labels.csv"
    if not csv_path.exists():
        return set()
    sources = set()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            source = row.get("source_path", "")
            if source:
                sources.add(str(Path(source).resolve()).lower())
    return sources


def update_summary(dataset_root: Path):
    import json

    summary = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "splits": {},
        "labels": CLASS_LABELS,
    }
    for split in SPLITS:
        summary["splits"][split] = {}
        for class_name in CLASS_LABELS:
            folder = dataset_root / split / class_name
            summary["splits"][split][class_name] = len(collect_images(folder)) if folder.exists() else 0

    summary_path = dataset_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def sanitize_prefix(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", text).strip("_")


def is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
