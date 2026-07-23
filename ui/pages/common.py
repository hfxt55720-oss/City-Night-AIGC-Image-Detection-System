from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget


def page_root(parent: QWidget) -> QVBoxLayout:
    layout = QVBoxLayout(parent)
    layout.setContentsMargins(28, 24, 28, 24)
    layout.setSpacing(18)
    return layout


def page_header(title: str, subtitle: str) -> QWidget:
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    title_label = QLabel(title)
    title_label.setObjectName("PageTitle")

    subtitle_label = QLabel(subtitle)
    subtitle_label.setObjectName("PageSubtitle")
    subtitle_label.setWordWrap(True)

    layout.addWidget(title_label)
    layout.addWidget(subtitle_label)
    return wrapper


def panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("Panel")
    frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 18)
    layout.setSpacing(12)

    title_label = QLabel(title)
    title_label.setObjectName("PanelTitle")
    layout.addWidget(title_label)
    return frame, layout


def metric_card(title: str, value: str, note: str = "") -> QFrame:
    frame = QFrame()
    frame.setObjectName("MetricCard")
    frame.setMinimumHeight(104)

    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(6)

    title_label = QLabel(title)
    title_label.setObjectName("MetricTitle")
    value_label = QLabel(value)
    value_label.setObjectName("MetricValue")
    note_label = QLabel(note)
    note_label.setObjectName("MutedText")
    note_label.setWordWrap(True)

    layout.addWidget(title_label)
    layout.addWidget(value_label)
    layout.addWidget(note_label)
    layout.addStretch()
    return frame


def empty_state(text: str, minimum_height: int = 180) -> QFrame:
    frame = QFrame()
    frame.setObjectName("EmptyState")
    frame.setMinimumHeight(minimum_height)
    frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    layout = QHBoxLayout(frame)
    layout.setContentsMargins(16, 16, 16, 16)

    label = QLabel(text)
    label.setObjectName("EmptyStateText")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)

    layout.addWidget(label)
    return frame
