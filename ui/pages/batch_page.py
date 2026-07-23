import csv
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from system.app_config import behavior_enabled, load_config, path_for
from system.history_store import append_many_history
from ui.pages.common import page_header, page_root, panel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "output" / "results"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class BatchPredictionWorker(QObject):
    row_ready = pyqtSignal(dict)
    progress_changed = pyqtSignal(int, int)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model, model_name: str, image_paths: list[Path]):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.image_paths = image_paths
        self.cancelled = False

    def stop(self):
        self.cancelled = True

    def run(self):
        try:
            total = len(self.image_paths)
            for index, image_path in enumerate(self.image_paths, start=1):
                if self.cancelled:
                    break

                row = {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "model_name": self.model_name,
                    "result": "ERROR",
                    "ai_probability": "",
                    "real_probability": "",
                    "time": "",
                    "error": "",
                }
                try:
                    prediction = self.model.predict(str(image_path))
                    row.update(
                        {
                            "result": prediction["result"],
                            "ai_probability": prediction["ai_probability"],
                            "real_probability": prediction["real_probability"],
                            "time": prediction["time"],
                        }
                    )
                except Exception as exc:
                    row["error"] = str(exc)

                self.row_ready.emit(row)
                self.progress_changed.emit(index, total)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class BatchPage(QWidget):
    def __init__(self):
        super().__init__()
        self.active_model = None
        self.active_model_name = "AIDE-Night"
        self.folder_path: Path | None = None
        self.image_paths: list[Path] = []
        self.rows: list[dict] = []
        self.worker: BatchPredictionWorker | None = None
        self.thread: QThread | None = None
        self.apply_settings(load_config())

        layout = page_root(self)
        layout.addWidget(page_header("批量检测", "选择图片文件夹，使用当前模型批量检测jpg、jpeg和png图片。"))

        task_panel, task_layout = panel("批量任务")

        folder_row = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("待检测图片文件夹")
        self.folder_input.setReadOnly(True)
        self.select_button = QPushButton("选择文件夹")
        self.select_button.setObjectName("SecondaryButton")
        self.select_button.clicked.connect(self.select_folder)
        folder_row.addWidget(self.folder_input, 1)
        folder_row.addWidget(self.select_button)

        action_row = QHBoxLayout()
        self.start_button = QPushButton("开始批量检测")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self.start_batch)
        self.export_button = QPushButton("导出CSV")
        self.export_button.setObjectName("SecondaryButton")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_csv)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.export_button)
        action_row.addStretch()

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.status_label = QLabel("AIDE-Night模型加载中...")
        self.status_label.setObjectName("MutedText")
        self.status_label.setWordWrap(True)

        task_layout.addLayout(folder_row)
        task_layout.addLayout(action_row)
        task_layout.addWidget(self.progress)
        task_layout.addWidget(self.status_label)

        result_panel, result_layout = panel("检测结果")
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["图片名称", "检测结果", "AI概率", "真实概率", "耗时"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 280)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(2, 100)
        self.table.setColumnWidth(3, 100)
        self.table.setColumnWidth(4, 90)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        result_layout.addWidget(self.table)

        page_splitter = QSplitter(Qt.Orientation.Vertical)
        page_splitter.setChildrenCollapsible(False)
        page_splitter.setHandleWidth(8)
        page_splitter.addWidget(task_panel)
        page_splitter.addWidget(result_panel)
        page_splitter.setStretchFactor(0, 0)
        page_splitter.setStretchFactor(1, 1)
        page_splitter.setSizes([170, 470])
        layout.addWidget(page_splitter, 1)

    def set_aide_model(self, model):
        self.set_active_model(model, "AIDE-Night", f"AIDE-Night模型已加载 | device={model.device}")

    def set_active_model(self, model, model_name: str, status_message: str | None = None):
        self.active_model = model
        self.active_model_name = model_name
        self.status_label.setText(status_message or f"当前模型：{model_name}")

    def set_model_unavailable(self, model_name: str, message: str):
        self.set_active_model(None, model_name, f"当前模型：{model_name}")
        self.status_label.setText(message)

    def set_model_error(self, message: str):
        self.active_model = None
        self.active_model_name = "AIDE-Night"
        self.status_label.setText(f"AIDE-Night模型加载失败：{message}")

    def apply_settings(self, config: dict):
        self.result_dir = path_for("results", config)
        self.save_history_enabled = behavior_enabled("save_history", config)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "")
        if not folder:
            return

        self.folder_path = Path(folder)
        self.folder_input.setText(str(self.folder_path))
        self.image_paths = self._collect_images(self.folder_path)
        self.status_label.setText(f"已找到 {len(self.image_paths)} 张图片 | 当前模型：{self.active_model_name}")
        self.progress.setValue(0)

    def start_batch(self):
        if self.active_model is None:
            QMessageBox.warning(self, "批量检测", f"{self.active_model_name}模型不可用或仍在加载。")
            return
        if not self.image_paths:
            QMessageBox.warning(self, "批量检测", "请先选择包含jpg、jpeg或png图片的文件夹。")
            return
        if self.thread is not None:
            QMessageBox.information(self, "批量检测", "批量检测正在进行中。")
            return

        self.rows = []
        self.table.setRowCount(0)
        self.progress.setRange(0, len(self.image_paths))
        self.progress.setValue(0)
        self.export_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.status_label.setText(f"正在使用{self.active_model_name}批量检测...")

        self.thread = QThread(self)
        self.worker = BatchPredictionWorker(self.active_model, self.active_model_name, self.image_paths)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.row_ready.connect(self._append_result_row)
        self.worker.progress_changed.connect(self._update_progress)
        self.worker.failed.connect(self._on_batch_failed)
        self.worker.finished.connect(self._on_batch_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._clear_worker)
        self.thread.start()

    def export_csv(self):
        if not self.rows:
            QMessageBox.information(self, "导出CSV", "当前没有可导出的检测结果。")
            return

        self.result_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = self.result_dir / f"batch_{timestamp}.csv"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出CSV",
            str(default_path),
            "CSV Files (*.csv)",
        )
        if not save_path:
            return

        with Path(save_path).open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "image_name",
                    "image_path",
                    "model_name",
                    "result",
                    "ai_probability",
                    "real_probability",
                    "time",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(self.rows)

        self.status_label.setText(f"CSV已导出：{save_path}")

    def _append_result_row(self, row: dict):
        self.rows.append(row)
        table_row = self.table.rowCount()
        self.table.insertRow(table_row)
        values = [
            row["image_name"],
            row["result"],
            self._format_probability(row["ai_probability"]),
            self._format_probability(row["real_probability"]),
            row["time"],
        ]
        for column, value in enumerate(values):
            self.table.setItem(table_row, column, QTableWidgetItem(str(value)))

    def _update_progress(self, current: int, total: int):
        self.progress.setValue(current)
        self.status_label.setText(f"检测进度：{current}/{total} | 当前模型：{self.active_model_name}")

    def _on_batch_failed(self, message: str):
        self.status_label.setText(f"批量检测失败：{message}")

    def _on_batch_finished(self):
        self.start_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.export_button.setEnabled(bool(self.rows))
        saved_records = [
            {
                "source": "batch",
                "image_name": row["image_name"],
                "image_path": row["image_path"],
                "model_name": row["model_name"],
                "result": row["result"],
                "ai_probability": row["ai_probability"],
                "real_probability": row["real_probability"],
                "time": row["time"],
            }
            for row in self.rows
            if row["result"] in {"AIGC", "REAL"}
        ]
        if self.save_history_enabled:
            append_many_history(saved_records)
        if self.rows:
            history_text = "已写入记录管理" if self.save_history_enabled else "未写入记录管理"
            self.status_label.setText(f"批量检测完成：{len(self.rows)} 张图片，{history_text}")
        else:
            self.status_label.setText("批量检测结束，未生成结果")

    def _clear_worker(self):
        self.thread = None
        self.worker = None

    def _collect_images(self, folder_path: Path) -> list[Path]:
        return [
            path
            for path in sorted(folder_path.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]

    def _format_probability(self, value):
        if value == "":
            return "-"
        return f"{float(value):.6f}"
