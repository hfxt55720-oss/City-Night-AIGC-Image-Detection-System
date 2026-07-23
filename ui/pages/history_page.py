import csv
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from models.model_registry import MODEL_SPECS
from system.app_config import display_path, path_for
from system.history_store import current_history_path, read_history, to_float, write_history
from ui.pages.common import page_header, page_root, panel
from ui.pages.detect_page import ImageViewerDialog


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif"}


def history_path_text() -> str:
    return display_path(current_history_path())


class ResultImageLabel(QLabel):
    def __init__(self, image_path: Path | None, title: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.title = title
        self.preview_height = 210
        self.setObjectName("HistoryImagePreview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(180, self.preview_height)
        self.setMaximumHeight(self.preview_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setWordWrap(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("点击放大查看")
        self._pixmap = QPixmap(str(image_path)) if image_path and image_path.exists() else QPixmap()
        self._update_preview()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.image_path and self.image_path.exists():
            dialog = ImageViewerDialog(self.image_path, self.title, self)
            dialog.exec()
            return
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview()

    def _update_preview(self):
        if self._pixmap.isNull():
            self.setText("图片不存在或未生成")
            return
        target_size = QSize(max(1, self.width() - 4), max(1, self.preview_height - 4))
        self.setPixmap(
            self._pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def sizeHint(self):
        return QSize(280, self.preview_height)

    def minimumSizeHint(self):
        return QSize(180, self.preview_height)


class HistoryDetailDialog(QDialog):
    def __init__(self, record: dict, parent=None):
        super().__init__(parent)
        self.record = record
        self.setObjectName("HistoryDetailDialog")
        self.setWindowTitle("检测结果详情")
        self.resize(1120, 820)
        self.setMinimumSize(860, 640)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        title = QLabel(f"检测结果详情 | {record.get('image_name', '-')}")
        title.setObjectName("PanelTitle")
        title.setWordWrap(True)
        root.addWidget(title)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("HistoryDetailScroll")
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("HistoryDetailContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        info_box = QFrame()
        info_box.setObjectName("ResultBox")
        info_grid = QGridLayout(info_box)
        info_grid.setContentsMargins(16, 16, 16, 16)
        info_grid.setHorizontalSpacing(12)
        info_grid.setVerticalSpacing(10)
        info_grid.setColumnMinimumWidth(0, 92)
        info_grid.setColumnStretch(0, 0)
        info_grid.setColumnStretch(1, 1)

        for row, (label_text, value) in enumerate(self._detail_fields(record)):
            name_label = QLabel(label_text)
            name_label.setObjectName("FieldLabel")
            name_label.setMinimumWidth(92)
            value_label = QLabel(str(value) if value not in (None, "") else "-")
            value_label.setObjectName("ResultValue")
            value_label.setWordWrap(True)
            value_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            info_grid.addWidget(name_label, row, 0)
            info_grid.addWidget(value_label, row, 1)
        layout.addWidget(info_box)

        image_grid = QGridLayout()
        image_grid.setHorizontalSpacing(12)
        image_grid.setVerticalSpacing(12)
        image_grid.setColumnStretch(0, 1)
        image_grid.setColumnStretch(1, 1)
        image_grid.setColumnStretch(2, 1)
        for column, (title_text, image_path) in enumerate(self._image_paths(record)):
            slot = QFrame()
            slot.setObjectName("ImageSlot")
            slot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            slot_layout = QVBoxLayout(slot)
            slot_layout.setContentsMargins(12, 12, 12, 12)
            slot_layout.setSpacing(8)

            label = QLabel(title_text)
            label.setObjectName("FieldLabel")
            image_label = ResultImageLabel(image_path, title_text, self)
            path_label = QLabel(str(image_path) if image_path else "无文件路径")
            path_label.setObjectName("MutedText")
            path_label.setWordWrap(True)
            path_label.setMaximumHeight(44)
            path_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            path_label.setToolTip(str(image_path) if image_path else "无文件路径")

            slot_layout.addWidget(label)
            slot_layout.addWidget(image_label)
            slot_layout.addWidget(path_label)
            image_grid.addWidget(slot, 0, column)
        layout.addLayout(image_grid)

        scroll_area.setWidget(content)
        root.addWidget(scroll_area, 1)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_button = QPushButton("关闭")
        close_button.setObjectName("SecondaryButton")
        close_button.clicked.connect(self.accept)
        close_row.addWidget(close_button)
        root.addLayout(close_row)

    def _detail_fields(self, record: dict) -> list[tuple[str, object]]:
        gradcam = record.get("gradcam") if isinstance(record.get("gradcam"), dict) else {}
        return [
            ("记录ID", record.get("id", "")),
            ("来源", "单张检测" if record.get("source") == "single" else "批量检测"),
            ("检测时间", record.get("timestamp", "")),
            ("图片名称", record.get("image_name", "")),
            ("图片路径", record.get("image_path", "")),
            ("结果文件", record.get("result_file", "")),
            ("所使用模型", record.get("model_name", "")),
            ("模型标识", record.get("model_key", "")),
            ("运行设备", record.get("device", "")),
            ("检测结果", record.get("result", "")),
            ("AIGC概率", self._format_probability(record.get("ai_probability"))),
            ("REAL概率", self._format_probability(record.get("real_probability"))),
            ("置信度", self._format_probability(record.get("confidence"))),
            ("推理耗时", record.get("time", "")),
            ("Grad-CAM层", gradcam.get("target_layer", "")),
            ("图片删除状态", "已删除" if record.get("images_deleted") else "未删除"),
            ("图片删除时间", record.get("images_deleted_at", "")),
            ("已删除图片", "\n".join(record.get("deleted_image_paths", []))),
            ("删除异常", self._format_delete_errors(record.get("image_delete_errors", {}))),
        ]

    def _image_paths(self, record: dict) -> list[tuple[str, Path | None]]:
        gradcam = self._resolved_gradcam(record)
        original_path = gradcam.get("original_path") or record.get("image_path")
        return [
            ("原图", self._path_or_none(original_path)),
            ("Grad-CAM热力图", self._path_or_none(gradcam.get("heatmap_path"))),
            ("叠加图", self._path_or_none(gradcam.get("overlay_path"))),
        ]

    def _resolved_gradcam(self, record: dict) -> dict:
        gradcam = record.get("gradcam") if isinstance(record.get("gradcam"), dict) else {}
        if gradcam.get("heatmap_path") or gradcam.get("overlay_path"):
            return gradcam

        inferred = self._infer_gradcam_paths(record)
        return inferred or gradcam

    def _infer_gradcam_paths(self, record: dict) -> dict:
        image_name = record.get("image_name") or record.get("image_path")
        if not image_name:
            return {}

        model_key = record.get("model_key") or self._model_key_from_name(record.get("model_name", ""))
        result = str(record.get("result", "")).lower()
        if not model_key or result not in {"aigc", "real"}:
            return {}

        stem = Path(str(image_name)).stem
        matches = sorted(
            path_for("heatmaps").glob(f"{stem}_{model_key}_{result}_*_overlay.jpg"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            return {}

        overlay_path = matches[0]
        base = str(overlay_path)[: -len("_overlay.jpg")]
        original_path = Path(f"{base}_original.jpg")
        heatmap_path = Path(f"{base}_heatmap.jpg")
        return {
            "original_path": str(original_path),
            "heatmap_path": str(heatmap_path),
            "overlay_path": str(overlay_path),
        }

    def _model_key_from_name(self, model_name: str) -> str:
        mapping = {
            "AIDE-Night": "aide",
            "ResNet50": "resnet50",
            "EfficientNet-B0": "efficientnet_b0",
        }
        return mapping.get(model_name, "")

    def _path_or_none(self, value) -> Path | None:
        if not value:
            return None
        return Path(str(value))

    def _format_probability(self, value) -> str:
        return f"{to_float(value):.6f}"

    def _format_delete_errors(self, value) -> str:
        if not isinstance(value, dict) or not value:
            return ""
        return "\n".join(f"{path}: {message}" for path, message in value.items())


class HistoryPage(QWidget):
    def __init__(self):
        super().__init__()
        self.records: list[dict] = []
        self.visible_records: list[dict] = []
        self.selection_checkboxes: dict[int, QCheckBox] = {}

        layout = page_root(self)
        layout.addWidget(page_header("记录管理", "按时间、模型、图片名称和结果检索记录，可查看详情或删除选中的记录。"))

        history_panel, history_layout = panel("历史记录")

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("多个关键词用 # 隔开，例如：2026-07-19#AIDE#AIGC")
        self.search_input.textChanged.connect(self.refresh_table)

        model_label = QLabel("模型")
        model_label.setObjectName("SelectorLabel")
        self.model_filter = QComboBox()
        self.model_filter.setObjectName("HistoryModelFilter")
        self.model_filter.currentIndexChanged.connect(self.refresh_table)

        self.refresh_button = QPushButton("刷新")
        self.refresh_button.setObjectName("SecondaryButton")
        self.refresh_button.clicked.connect(self.load_records)

        self.clear_filter_button = QPushButton("清空筛选")
        self.clear_filter_button.setObjectName("SecondaryButton")
        self.clear_filter_button.clicked.connect(self.clear_filters)

        search_row.addWidget(self.search_input, 1)
        search_row.addWidget(model_label)
        search_row.addWidget(self.model_filter)
        search_row.addWidget(self.refresh_button)
        search_row.addWidget(self.clear_filter_button)

        action_row = QHBoxLayout()
        self.select_all_button = QPushButton("一键勾选")
        self.select_all_button.setObjectName("SecondaryButton")
        self.select_all_button.clicked.connect(self.check_all_visible)

        self.clear_checked_button = QPushButton("取消勾选")
        self.clear_checked_button.setObjectName("SecondaryButton")
        self.clear_checked_button.clicked.connect(self.clear_checked)

        self.delete_button = QPushButton("删除选中图片")
        self.delete_button.setObjectName("SecondaryButton")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.delete_checked_records)

        self.export_button = QPushButton("导出记录")
        self.export_button.setObjectName("SecondaryButton")
        self.export_button.clicked.connect(self.export_records)

        action_row.addWidget(self.select_all_button)
        action_row.addWidget(self.clear_checked_button)
        action_row.addWidget(self.delete_button)
        action_row.addStretch()
        action_row.addWidget(self.export_button)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["时间", "图片名称", "模型", "结果", "AIGC概率", "置信度", "查看", "选择"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(1, 220)
        self.table.setColumnWidth(2, 140)
        self.table.setColumnWidth(3, 80)
        self.table.setColumnWidth(4, 100)
        self.table.setColumnWidth(5, 90)
        self.table.setColumnWidth(6, 80)
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(46)
        self.table.verticalHeader().setMinimumSectionSize(46)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self.open_row_detail)

        self.status_label = QLabel()
        self.status_label.setObjectName("MutedText")
        self.status_label.setWordWrap(True)

        history_layout.addLayout(search_row)
        history_layout.addLayout(action_row)
        history_layout.addWidget(self.table, 1)
        history_layout.addWidget(self.status_label)
        layout.addWidget(history_panel, 1)

        self.load_records()

    def load_records(self):
        raw_records = read_history()
        self.records = []
        for history_index, record in enumerate(raw_records):
            tagged_record = dict(record)
            tagged_record["_history_index"] = history_index
            self.records.append(tagged_record)
        self.records.reverse()
        self.refresh_model_filter()
        self.refresh_table()

    def refresh_model_filter(self):
        current_model = self.model_filter.currentData() if self.model_filter.count() else ""
        model_names = {spec["name"] for spec in MODEL_SPECS.values()}
        model_names.update(str(record.get("model_name", "")).strip() for record in self.records if record.get("model_name"))

        self.model_filter.blockSignals(True)
        self.model_filter.clear()
        self.model_filter.addItem("全部模型", "")
        for name in sorted(model_names):
            self.model_filter.addItem(name, name)

        index = self.model_filter.findData(current_model)
        self.model_filter.setCurrentIndex(index if index >= 0 else 0)
        self.model_filter.blockSignals(False)

    def refresh_table(self, *args):
        keywords = self._search_keywords()
        selected_model = self.model_filter.currentData() if self.model_filter.count() else ""

        self.visible_records = [
            record
            for record in self.records
            if self._record_matches(record, keywords, selected_model)
        ]

        self.selection_checkboxes = {}
        self.table.setRowCount(0)
        for record in self.visible_records:
            self._append_table_row(record)

        total = len(self.records)
        visible = len(self.visible_records)
        selected = len(self._selected_history_indices())
        source_text = f"记录文件：{history_path_text()}"
        if total == 0:
            self.status_label.setText(f"暂无记录。{source_text}")
        elif visible != total:
            self.status_label.setText(f"显示 {visible}/{total} 条记录，已勾选 {selected} 条。{source_text}")
        else:
            self.status_label.setText(f"共 {total} 条记录，已勾选 {selected} 条。{source_text}")
        self.delete_button.setEnabled(selected > 0)

    def clear_filters(self):
        self.search_input.clear()
        index = self.model_filter.findData("")
        if index >= 0:
            self.model_filter.setCurrentIndex(index)
        self.clear_checked()

    def check_all_visible(self):
        for checkbox in self.selection_checkboxes.values():
            checkbox.setChecked(True)
        self._update_selection_status()

    def clear_checked(self):
        for checkbox in self.selection_checkboxes.values():
            checkbox.setChecked(False)
        self._update_selection_status()

    def delete_checked_records(self):
        selected_indices = self._selected_history_indices()
        if not selected_indices:
            QMessageBox.information(self, "删除图片", "请先勾选需要删除图片的记录。")
            return

        selected_records = [
            record
            for record in self.visible_records
            if int(record.get("_history_index", -1)) in selected_indices
        ]
        image_paths = self._collect_image_paths(selected_records)
        delete_mode = self._confirm_delete_choice(len(selected_records), len(image_paths))
        if delete_mode is None:
            return

        deleted_paths, missing_paths, failed_paths = self._delete_image_files(image_paths)
        raw_records = read_history()
        if delete_mode == "delete_records":
            raw_records = [
                record
                for history_index, record in enumerate(raw_records)
                if history_index not in selected_indices
            ]
        else:
            deleted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for history_index in selected_indices:
                if 0 <= history_index < len(raw_records):
                    raw_records[history_index] = self._mark_record_images_deleted(
                        raw_records[history_index],
                        deleted_at,
                        deleted_paths,
                        missing_paths,
                        failed_paths,
                    )

        write_history(raw_records)
        self.load_records()
        record_text = "记录已同时删除" if delete_mode == "delete_records" else "记录已保留"
        self.status_label.setText(
            f"已删除 {len(deleted_paths)} 个图片文件，{record_text}。"
            f"缺失 {len(missing_paths)} 个，失败 {len(failed_paths)} 个。记录文件：{history_path_text()}"
        )

    def _confirm_delete_choice(self, record_count: int, image_count: int) -> str | None:
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("删除图片")
        message.setText(f"是否删除选中 {record_count} 条记录对应的 {image_count} 个图片文件？")
        message.setInformativeText(
            "将删除原图、Grad-CAM热力图和叠加图等关联图片。默认保留记录；也可以选择把记录一起删除。此操作不可恢复。"
        )
        keep_record_button = message.addButton("删除图片，保留记录", QMessageBox.ButtonRole.AcceptRole)
        delete_record_button = message.addButton("删除图片和记录", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = message.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        message.setDefaultButton(cancel_button)
        message.exec()

        clicked_button = message.clickedButton()
        if clicked_button == keep_record_button:
            return "keep_records"
        if clicked_button == delete_record_button:
            return "delete_records"
        return None

    def _collect_image_paths(self, records: list[dict]) -> list[Path]:
        unique_paths: dict[str, Path] = {}
        for record in records:
            for _, image_path in self._image_paths(record):
                if image_path is None or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                unique_paths[self._path_key(image_path)] = image_path
        return list(unique_paths.values())

    def _delete_image_files(self, image_paths: list[Path]) -> tuple[list[str], list[str], dict[str, str]]:
        deleted_paths = []
        missing_paths = []
        failed_paths = {}

        for image_path in image_paths:
            path_text = str(image_path)
            try:
                if not image_path.exists():
                    missing_paths.append(path_text)
                    continue
                if not image_path.is_file():
                    failed_paths[path_text] = "路径不是文件"
                    continue
                image_path.unlink()
                deleted_paths.append(path_text)
            except Exception as exc:
                failed_paths[path_text] = str(exc)

        return deleted_paths, missing_paths, failed_paths

    def _mark_record_images_deleted(
        self,
        record: dict,
        deleted_at: str,
        deleted_paths: list[str],
        missing_paths: list[str],
        failed_paths: dict[str, str],
    ) -> dict:
        updated = dict(record)
        own_paths = {str(path) for path in self._collect_image_paths([record])}
        deleted_set = set(deleted_paths)
        missing_set = set(missing_paths)
        failed_set = set(failed_paths)

        previous_deleted = set(updated.get("deleted_image_paths", []))
        previous_missing = set(updated.get("missing_image_paths", []))
        previous_errors = dict(updated.get("image_delete_errors", {}))

        updated["images_deleted"] = True
        updated["images_deleted_at"] = deleted_at
        updated["deleted_image_paths"] = sorted(previous_deleted | (own_paths & deleted_set))
        updated["missing_image_paths"] = sorted(previous_missing | (own_paths & missing_set))
        previous_errors.update({path: failed_paths[path] for path in sorted(own_paths & failed_set)})
        updated["image_delete_errors"] = previous_errors
        return updated

    def _path_key(self, image_path: Path) -> str:
        try:
            return str(image_path.resolve(strict=False)).casefold()
        except Exception:
            return str(image_path).casefold()

    def export_records(self):
        if not self.visible_records:
            QMessageBox.information(self, "导出记录", "当前没有可导出的记录。")
            return

        result_dir = path_for("results")
        result_dir.mkdir(parents=True, exist_ok=True)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出记录管理",
            str(result_dir / "history_export.csv"),
            "CSV Files (*.csv)",
        )
        if not save_path:
            return

        fieldnames = [
            "id",
            "timestamp",
            "source",
            "image_name",
            "image_path",
            "model_key",
            "model_name",
            "device",
            "result",
            "ai_probability",
            "real_probability",
            "confidence",
            "time",
            "result_file",
            "images_deleted",
            "images_deleted_at",
            "deleted_image_paths",
            "missing_image_paths",
            "image_delete_errors",
        ]
        with Path(save_path).open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.visible_records)

        self.status_label.setText(f"记录已导出：{save_path}")

    def open_row_detail(self, row: int, column: int = 0):
        if 0 <= row < len(self.visible_records):
            dialog = HistoryDetailDialog(self.visible_records[row], self)
            dialog.exec()

    def _append_table_row(self, record: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)

        values = [
            record.get("timestamp", "-"),
            record.get("image_name", "-"),
            record.get("model_name", "-"),
            record.get("result", "-"),
            self._format_probability(record.get("ai_probability")),
            self._format_probability(record.get("confidence")),
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self.table.setItem(row, column, item)

        view_button = QPushButton("查看结果")
        view_button.setObjectName("TableActionButton")
        view_button.setFixedHeight(30)
        view_button.clicked.connect(lambda checked=False, row_index=row: self.open_row_detail(row_index))
        view_cell = QWidget()
        view_cell.setObjectName("HistoryActionCell")
        view_layout = QHBoxLayout(view_cell)
        view_layout.setContentsMargins(4, 4, 4, 4)
        view_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        view_layout.addWidget(view_button)
        self.table.setCellWidget(row, 6, view_cell)

        history_index = int(record.get("_history_index", -1))
        checkbox = QCheckBox()
        checkbox.stateChanged.connect(self._update_selection_status)
        checkbox_widget = QWidget()
        checkbox_widget.setObjectName("HistorySelectionCell")
        checkbox_layout = QHBoxLayout(checkbox_widget)
        checkbox_layout.setContentsMargins(4, 4, 4, 4)
        checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        checkbox_layout.addWidget(checkbox)
        self.selection_checkboxes[history_index] = checkbox
        self.table.setCellWidget(row, 7, checkbox_widget)
        self.table.setRowHeight(row, 46)

    def _image_paths(self, record: dict) -> list[tuple[str, Path | None]]:
        gradcam = self._resolved_gradcam(record)
        original_path = gradcam.get("original_path") or record.get("image_path")
        return [
            ("原图", self._path_or_none(original_path)),
            ("Grad-CAM热力图", self._path_or_none(gradcam.get("heatmap_path"))),
            ("叠加图", self._path_or_none(gradcam.get("overlay_path"))),
        ]

    def _resolved_gradcam(self, record: dict) -> dict:
        gradcam = record.get("gradcam") if isinstance(record.get("gradcam"), dict) else {}
        if gradcam.get("heatmap_path") or gradcam.get("overlay_path"):
            return gradcam

        inferred = self._infer_gradcam_paths(record)
        return inferred or gradcam

    def _infer_gradcam_paths(self, record: dict) -> dict:
        image_name = record.get("image_name") or record.get("image_path")
        if not image_name:
            return {}

        model_key = record.get("model_key") or self._model_key_from_name(record.get("model_name", ""))
        result = str(record.get("result", "")).lower()
        if not model_key or result not in {"aigc", "real"}:
            return {}

        stem = Path(str(image_name)).stem
        matches = sorted(
            path_for("heatmaps").glob(f"{stem}_{model_key}_{result}_*_overlay.jpg"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            return {}

        overlay_path = matches[0]
        base = str(overlay_path)[: -len("_overlay.jpg")]
        original_path = Path(f"{base}_original.jpg")
        heatmap_path = Path(f"{base}_heatmap.jpg")
        return {
            "original_path": str(original_path),
            "heatmap_path": str(heatmap_path),
            "overlay_path": str(overlay_path),
        }

    def _model_key_from_name(self, model_name: str) -> str:
        mapping = {
            "AIDE-Night": "aide",
            "ResNet50": "resnet50",
            "EfficientNet-B0": "efficientnet_b0",
        }
        return mapping.get(model_name, "")

    def _path_or_none(self, value) -> Path | None:
        if not value:
            return None
        return Path(str(value))

    def _record_matches(self, record: dict, keywords: list[str], selected_model: str) -> bool:
        if selected_model and record.get("model_name") != selected_model:
            return False
        if not keywords:
            return True

        searchable_values = [
            record.get("timestamp", ""),
            record.get("model_name", ""),
            record.get("image_name", ""),
            record.get("result", ""),
        ]
        searchable_text = " ".join(str(value).lower() for value in searchable_values)
        return all(keyword in searchable_text for keyword in keywords)

    def _search_keywords(self) -> list[str]:
        text = self.search_input.text().strip().lower()
        if not text:
            return []
        return [keyword.strip() for keyword in text.split("#") if keyword.strip()]

    def _selected_history_indices(self) -> set[int]:
        return {
            history_index
            for history_index, checkbox in self.selection_checkboxes.items()
            if history_index >= 0 and checkbox.isChecked()
        }

    def _update_selection_status(self, *args):
        selected = len(self._selected_history_indices())
        visible = len(self.visible_records)
        self.delete_button.setEnabled(selected > 0)
        self.status_label.setText(f"当前筛选显示 {visible} 条记录，已勾选 {selected} 条。记录文件：{history_path_text()}")

    def _format_probability(self, value) -> str:
        return f"{to_float(value):.6f}"
