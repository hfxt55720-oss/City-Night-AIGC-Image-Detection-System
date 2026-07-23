from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from models.model_registry import MODEL_SPECS, SUPPORTED_MODEL_TYPES, register_custom_model
from system.app_config import APP_ROOT, RESOURCE_ROOT
from ui.pages.common import page_header, page_root, panel


def display_model_path(path) -> str:
    path = Path(path)
    roots = [APP_ROOT]
    if RESOURCE_ROOT != APP_ROOT:
        roots.append(RESOURCE_ROOT)
    for root in roots:
        try:
            return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
        except Exception:
            continue
    return str(path)


class AddModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加模型")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        note = QLabel("添加模型需要选择兼容的模型结构和对应权重文件。全新网络结构需要先开发推理封装。")
        note.setObjectName("MutedText")
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setSpacing(10)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如 AIDE-Night-v2")

        self.arch_combo = QComboBox()
        for key, name in SUPPORTED_MODEL_TYPES.items():
            self.arch_combo.addItem(name, key)

        self.weight_edit = QLineEdit()
        self.weight_edit.setPlaceholderText("选择 .pth 或 .pt 权重文件")
        browse_button = QPushButton("浏览")
        browse_button.setObjectName("SecondaryButton")
        browse_button.setFixedWidth(76)
        browse_button.clicked.connect(self.choose_weight)
        weight_row = QHBoxLayout()
        weight_row.setContentsMargins(0, 0, 0, 0)
        weight_row.setSpacing(8)
        weight_row.addWidget(self.weight_edit, 1)
        weight_row.addWidget(browse_button)

        form.addRow("显示名称", self.name_edit)
        form.addRow("模型结构", self.arch_combo)
        form.addRow("权重文件", weight_row)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("添加")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def choose_weight(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择权重文件", "", "Weights (*.pth *.pt);;All Files (*)")
        if file_path:
            self.weight_edit.setText(file_path)

    def values(self) -> tuple[str, str, str]:
        return self.name_edit.text().strip(), str(self.arch_combo.currentData()), self.weight_edit.text().strip()


class ModelPage(QWidget):
    model_selected = pyqtSignal(str)
    model_added = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.model_keys = list(MODEL_SPECS.keys())
        self.current_model_key = "aide"
        self.selected_model_key = "aide"
        self.runtime_status = {key: ("加载中" if key == "aide" else "未加载") for key in self.model_keys}
        self.card_buttons: dict[str, QPushButton] = {}

        layout = page_root(self)
        layout.addWidget(page_header("模型管理", "点击模型卡片或表格行只会选中模型，点击“使用选中模型”后才会切换当前检测模型。"))

        page_splitter = QSplitter(Qt.Orientation.Vertical)
        page_splitter.setChildrenCollapsible(False)
        page_splitter.setHandleWidth(8)

        card_container = QWidget()
        self.card_row = QHBoxLayout(card_container)
        self.card_row.setContentsMargins(0, 0, 0, 0)
        self.card_row.setSpacing(14)
        for key in self.model_keys:
            button = self._build_model_card(key)
            self.card_buttons[key] = button
            self.card_row.addWidget(button)
        page_splitter.addWidget(card_container)

        model_panel, model_layout = panel("模型列表")

        self.current_label = QLabel()
        self.current_label.setObjectName("MutedText")
        model_layout.addWidget(self.current_label)

        self.table = QTableWidget(len(self.model_keys), 4)
        self.table.setHorizontalHeaderLabels(["模型名称", "模型路径", "当前使用模型", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 180)
        self.table.setColumnWidth(1, 520)
        self.table.setColumnWidth(2, 110)
        self.table.setColumnWidth(3, 100)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.cellClicked.connect(self._on_table_cell_clicked)
        self.table.cellDoubleClicked.connect(self._on_table_cell_clicked)
        self.table.setAlternatingRowColors(True)
        for row in range(len(self.model_keys)):
            self.table.setRowHeight(row, 48)

        button_row = QHBoxLayout()
        add_button = QPushButton("添加模型")
        add_button.setObjectName("PrimaryButton")
        add_button.clicked.connect(self.add_model)
        refresh_button = QPushButton("刷新状态")
        refresh_button.setObjectName("SecondaryButton")
        refresh_button.clicked.connect(self.refresh_table)
        self.switch_button = QPushButton("使用选中模型")
        self.switch_button.setObjectName("PrimaryButton")
        self.switch_button.clicked.connect(self.switch_selected_model)
        button_row.addWidget(add_button)
        button_row.addStretch()
        button_row.addWidget(refresh_button)
        button_row.addWidget(self.switch_button)

        model_layout.addWidget(self.table)
        model_layout.addLayout(button_row)
        page_splitter.addWidget(model_panel)
        page_splitter.setStretchFactor(0, 0)
        page_splitter.setStretchFactor(1, 1)
        page_splitter.setSizes([170, 480])
        layout.addWidget(page_splitter, 1)

        self.refresh_table()
        self.table.selectRow(0)

    def set_current_model(self, model_key: str):
        if model_key not in MODEL_SPECS:
            return
        self.current_model_key = model_key
        self.selected_model_key = model_key
        self._sync_model_keys()
        self.refresh_table()

    def set_model_status(self, model_key: str, status: str):
        if model_key not in MODEL_SPECS:
            return
        self.runtime_status[model_key] = status
        self.refresh_table()

    def switch_selected_model(self):
        model_key = self.selected_model_key
        row = self.table.currentRow()
        if model_key not in MODEL_SPECS and 0 <= row < len(self.model_keys):
            model_key = self.model_keys[row]
        if model_key not in MODEL_SPECS:
            return

        self.current_model_key = model_key
        self.selected_model_key = model_key
        self.refresh_table()
        self.model_selected.emit(model_key)

    def add_model(self):
        dialog = AddModelDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        name, architecture, weight_path = dialog.values()
        if not name:
            QMessageBox.warning(self, "添加模型", "请输入模型显示名称。")
            return
        if not weight_path:
            QMessageBox.warning(self, "添加模型", "请选择权重文件。")
            return

        try:
            model_key = register_custom_model(name, architecture, weight_path)
        except Exception as exc:
            QMessageBox.warning(self, "添加模型失败", str(exc))
            return

        self.runtime_status[model_key] = "未加载"
        self.selected_model_key = model_key
        self._sync_model_keys()
        self.refresh_table()
        self.model_added.emit(model_key)
        QMessageBox.information(self, "添加模型", f"模型已添加并选中：\n{MODEL_SPECS[model_key]['name']}\n\n点击“使用选中模型”后才会切换为当前检测模型。")

    def refresh_table(self):
        self._sync_model_keys()
        selected_key = self.selected_model_key if self.selected_model_key in MODEL_SPECS else self.current_model_key
        self.table.setRowCount(len(self.model_keys))
        for row, key in enumerate(self.model_keys):
            self.table.setRowHeight(row, 48)
            spec = MODEL_SPECS[key]
            path = spec["weight_path"]
            file_exists = path.exists()
            status = self.runtime_status.get(key, "未知")
            if not file_exists:
                status = "权重缺失"

            values = [
                spec["name"],
                display_model_path(path),
                self._current_column_text(key),
                status,
            ]
            for column, value in enumerate(values):
                self.table.setItem(
                    row,
                    column,
                    self._table_item(
                        value,
                        current=key == self.current_model_key,
                        selected=key == selected_key,
                    ),
                )

            self.card_buttons[key].setChecked(key == selected_key)
            self.card_buttons[key].setText(self._card_text(key))
            if key == selected_key:
                self.table.selectRow(row)

        current_name = MODEL_SPECS[self.current_model_key]["name"]
        selected_name = MODEL_SPECS[selected_key]["name"] if selected_key in MODEL_SPECS else current_name
        if selected_key == self.current_model_key:
            self.current_label.setText(f"当前使用模型：{current_name}")
        else:
            self.current_label.setText(f"当前使用模型：{current_name} | 已选中：{selected_name}，点击“使用选中模型”后生效")

    def _build_model_card(self, model_key: str) -> QPushButton:
        button = QPushButton(self._card_text(model_key))
        button.setObjectName("ModelCardButton")
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda checked=False, key=model_key: self._preview_model(key))
        return button

    def _sync_model_keys(self):
        current_keys = list(MODEL_SPECS.keys())
        if current_keys == self.model_keys and all(key in self.card_buttons for key in current_keys):
            return

        self.model_keys = current_keys
        if self.selected_model_key not in MODEL_SPECS:
            self.selected_model_key = self.current_model_key
        for key in self.model_keys:
            self.runtime_status.setdefault(key, "未加载")
            if key in self.card_buttons:
                continue
            button = self._build_model_card(key)
            self.card_buttons[key] = button
            self.card_row.addWidget(button)

        for key in list(self.card_buttons):
            if key in self.model_keys:
                continue
            button = self.card_buttons.pop(key)
            self.card_row.removeWidget(button)
            button.deleteLater()

    def _card_text(self, model_key: str) -> str:
        spec = MODEL_SPECS[model_key]
        file_state = "权重存在" if spec["weight_path"].exists() else "权重缺失"
        if model_key == self.current_model_key:
            current = "当前使用"
        elif model_key == self.selected_model_key:
            current = "已选中，点击按钮使用"
        else:
            current = "点击选中"
        status = self.runtime_status.get(model_key, "未知")
        return f"{spec['name']}\n{file_state}\n{status} | {current}\n{spec['weight_file']}"

    def _on_table_cell_clicked(self, row: int, column: int):
        if 0 <= row < len(self.model_keys):
            self._preview_model(self.model_keys[row])

    def _preview_model(self, model_key: str):
        if model_key not in MODEL_SPECS:
            return
        self.selected_model_key = model_key
        self.refresh_table()

    def _current_column_text(self, model_key: str) -> str:
        if model_key == self.current_model_key:
            return "当前使用"
        if model_key == self.selected_model_key:
            return "已选中"
        return "否"

    def _table_item(self, text: str, current: bool = False, selected: bool = False) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setForeground(QBrush(QColor("#e8edf3")))
        if current:
            item.setBackground(QBrush(QColor("#172a46")))
        elif selected:
            item.setBackground(QBrush(QColor("#2a2414")))
        item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        return item
