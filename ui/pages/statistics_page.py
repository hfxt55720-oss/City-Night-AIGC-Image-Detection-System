from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from system.app_config import display_path
from system.history_store import current_history_path, max_probability, normalize_result, read_history, summarize_history, to_float
from ui.pages.common import page_header, page_root, panel


def history_path_text() -> str:
    return display_path(current_history_path())


class MetricBox(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("MetricCard")
        self.setMinimumHeight(92)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("MetricTitle")
        self.value_label = QLabel("0")
        self.value_label.setObjectName("MetricValue")

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        layout.addStretch()

    def set_value(self, value: str):
        self.value_label.setText(value)


class PieChartWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.aigc_count = 0
        self.real_count = 0
        self.setMinimumHeight(190)
        self.setMaximumHeight(230)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_counts(self, aigc_count: int, real_count: int):
        self.aigc_count = aigc_count
        self.real_count = real_count
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        total = self.aigc_count + self.real_count
        if total == 0:
            self._draw_empty(painter)
            return

        side = min(self.width() - 180, self.height() - 36)
        side = max(side, 120)
        rect = QRectF(24, 18, side, side)

        aigc_angle = int(360 * self.aigc_count / total * 16)
        real_angle = 360 * 16 - aigc_angle

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2d74da"))
        painter.drawPie(rect, 90 * 16, -aigc_angle)
        painter.setBrush(QColor("#22a06b"))
        painter.drawPie(rect, 90 * 16 - aigc_angle, -real_angle)

        painter.setPen(QPen(QColor("#e8edf3")))
        painter.setFont(QFont("Microsoft YaHei UI", 10))
        legend_x = rect.right() + 28
        self._draw_legend(painter, legend_x, 42, "#2d74da", "AIGC", self.aigc_count, total)
        self._draw_legend(painter, legend_x, 84, "#22a06b", "REAL", self.real_count, total)

    def _draw_empty(self, painter: QPainter):
        painter.setPen(QPen(QColor("#9aa6b2")))
        painter.setFont(QFont("Microsoft YaHei UI", 11))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "暂无记录")

    def _draw_legend(self, painter: QPainter, x: float, y: float, color: str, name: str, count: int, total: int):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(int(x), int(y), 14, 14, 3, 3)
        painter.setPen(QPen(QColor("#dce3ec")))
        percent = count / total * 100 if total else 0
        painter.drawText(int(x + 24), int(y + 13), f"{name}: {count} ({percent:.1f}%)")


class BarChartWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.aigc_count = 0
        self.real_count = 0
        self.setMinimumHeight(190)
        self.setMaximumHeight(230)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_counts(self, aigc_count: int, real_count: int):
        self.aigc_count = aigc_count
        self.real_count = real_count
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        values = [("AIGC", self.aigc_count, QColor("#2d74da")), ("REAL", self.real_count, QColor("#22a06b"))]
        max_value = max([value for _, value, _ in values] + [1])

        label_height = 22
        label_gap = 8
        left = 52
        right = self.width() - 30
        bottom = self.height() - 46
        top = 54
        chart_height = max(1, bottom - top)
        bar_width = min(86, max(42, (right - left) // 5))
        gap = max(42, (right - left - bar_width * 2) // 3)

        painter.setPen(QPen(QColor("#354254"), 1))
        painter.drawLine(left, bottom, right, bottom)
        painter.drawLine(left, top, left, bottom)

        value_font = QFont("Microsoft YaHei UI", 10)
        value_font.setBold(True)
        painter.setFont(value_font)
        for index, (name, value, color) in enumerate(values):
            x = left + gap + index * (bar_width + gap)
            bar_height = int(chart_height * value / max_value) if max_value else 0
            y = bottom - bar_height

            if bar_height > 0:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawRoundedRect(x, y, bar_width, bar_height, 5, 5)

            painter.setPen(QPen(QColor("#e8edf3")))
            value_text = str(value)
            text_width = max(bar_width + 24, painter.fontMetrics().horizontalAdvance(value_text) + 18)
            text_x = int(x + bar_width / 2 - text_width / 2)
            text_x = max(0, min(text_x, self.width() - text_width))
            text_y = max(4, y - label_gap - label_height)
            painter.drawText(
                text_x,
                text_y,
                text_width,
                label_height,
                Qt.AlignmentFlag.AlignCenter,
                value_text,
            )

            painter.setFont(QFont("Microsoft YaHei UI", 10))
            painter.drawText(x - 10, bottom + 8, bar_width + 20, 24, Qt.AlignmentFlag.AlignCenter, name)
            painter.setFont(value_font)


class StatisticsPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = page_root(self)

        header_row = QHBoxLayout()
        header_row.addWidget(page_header("数据统计", "从本地JSON记录管理数据读取内容，统计检测数量、类别分布和平均置信度。"), 1)
        refresh_button = QPushButton("刷新统计")
        refresh_button.setObjectName("SecondaryButton")
        refresh_button.clicked.connect(self.refresh_statistics)
        header_row.addWidget(refresh_button)
        layout.addLayout(header_row)

        cards = QHBoxLayout()
        cards.setSpacing(14)
        self.total_metric = MetricBox("总检测数量")
        self.aigc_metric = MetricBox("AIGC数量")
        self.real_metric = MetricBox("REAL数量")
        self.confidence_metric = MetricBox("平均置信度")
        cards.addWidget(self.total_metric)
        cards.addWidget(self.aigc_metric)
        cards.addWidget(self.real_metric)
        cards.addWidget(self.confidence_metric)
        layout.addLayout(cards)

        content_splitter = QSplitter(Qt.Orientation.Vertical)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.setHandleWidth(8)

        chart_splitter = QSplitter(Qt.Orientation.Horizontal)
        chart_splitter.setChildrenCollapsible(False)
        chart_splitter.setHandleWidth(8)

        pie_panel, pie_layout = panel("类别占比")
        self.pie_chart = PieChartWidget()
        pie_layout.addWidget(self.pie_chart)

        bar_panel, bar_layout = panel("类别数量")
        self.bar_chart = BarChartWidget()
        bar_layout.addWidget(self.bar_chart)

        chart_splitter.addWidget(pie_panel)
        chart_splitter.addWidget(bar_panel)
        chart_splitter.setStretchFactor(0, 1)
        chart_splitter.setStretchFactor(1, 1)
        chart_splitter.setSizes([1, 1])
        content_splitter.addWidget(chart_splitter)

        table_panel, table_layout = panel("模型检测数量")
        table_panel.setMinimumHeight(230)
        table_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.model_table = QTableWidget(0, 7)
        self.model_table.setMinimumHeight(168)
        self.model_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.model_table.setHorizontalHeaderLabels(
            ["模型名称", "检测总数", "占比", "AIGC数量", "REAL数量", "平均置信度", "最近检测时间"]
        )
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.model_table.horizontalHeader().setStretchLastSection(False)
        self.model_table.setColumnWidth(0, 210)
        self.model_table.setColumnWidth(1, 82)
        self.model_table.setColumnWidth(2, 70)
        self.model_table.setColumnWidth(3, 88)
        self.model_table.setColumnWidth(4, 88)
        self.model_table.setColumnWidth(5, 104)
        self.model_table.setColumnWidth(6, 150)
        self.model_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.model_table, 1)

        self.source_label = QLabel()
        self.source_label.setObjectName("MutedText")
        table_layout.addWidget(self.source_label)
        content_splitter.addWidget(table_panel)
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([260, 320])
        layout.addWidget(content_splitter, 1)

        self.refresh_statistics()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_statistics()

    def refresh_statistics(self):
        records = read_history()
        summary = summarize_history(records)

        self.total_metric.set_value(str(summary["total"]))
        self.aigc_metric.set_value(str(summary["aigc_count"]))
        self.real_metric.set_value(str(summary["real_count"]))
        self.confidence_metric.set_value(f"{summary['avg_confidence'] * 100:.2f}%")

        self.pie_chart.set_counts(summary["aigc_count"], summary["real_count"])
        self.bar_chart.set_counts(summary["aigc_count"], summary["real_count"])
        self._refresh_model_table(records)
        self.source_label.setText(f"记录文件：{history_path_text()}")

    def _refresh_model_table(self, records: list[dict]):
        rows = build_model_statistics(records)
        self.model_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            values = [
                item["model_name"],
                str(item["total"]),
                f"{item['ratio'] * 100:.1f}%",
                str(item["aigc_count"]),
                str(item["real_count"]),
                f"{item['avg_confidence'] * 100:.2f}%",
                item["latest_time"] or "-",
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                table_item.setToolTip(value)
                if column > 0:
                    table_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                else:
                    table_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                self.model_table.setItem(row, column, table_item)
            self.model_table.setRowHeight(row, 38)

        if not rows:
            self.model_table.setRowCount(1)
            empty_item = QTableWidgetItem("暂无模型检测记录")
            empty_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.model_table.setItem(0, 0, empty_item)
            self.model_table.setSpan(0, 0, 1, self.model_table.columnCount())
        else:
            self.model_table.clearSpans()


def build_model_statistics(records: list[dict]) -> list[dict]:
    total_records = len(records)
    stats: dict[str, dict] = {}
    for record in records:
        model_name = str(record.get("model_name") or "未知模型")
        item = stats.setdefault(
            model_name,
            {
                "model_name": model_name,
                "total": 0,
                "aigc_count": 0,
                "real_count": 0,
                "confidence_sum": 0.0,
                "latest_time": "",
            },
        )
        item["total"] += 1

        result = normalize_result(record.get("result"))
        if result == "AIGC":
            item["aigc_count"] += 1
        elif result == "REAL":
            item["real_count"] += 1

        item["confidence_sum"] += to_float(record.get("confidence", max_probability(record)))
        timestamp = str(record.get("timestamp") or record.get("saved_at") or "")
        if timestamp and timestamp > item["latest_time"]:
            item["latest_time"] = timestamp

    rows = []
    for item in stats.values():
        total = item["total"]
        rows.append(
            {
                "model_name": item["model_name"],
                "total": total,
                "ratio": total / total_records if total_records else 0.0,
                "aigc_count": item["aigc_count"],
                "real_count": item["real_count"],
                "avg_confidence": item["confidence_sum"] / total if total else 0.0,
                "latest_time": item["latest_time"],
            }
        )
    return sorted(rows, key=lambda item: (-item["total"], item["model_name"]))
