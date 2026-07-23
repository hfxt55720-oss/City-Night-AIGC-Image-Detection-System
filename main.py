import sys

from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import QApplication

from system.app_config import resource_path
from ui.main_window import MainWindow


APP_NAME = "城市夜景AIGC图像检测系统"
APP_ICON_PATH = resource_path("data/img/tup_1.png")


def set_windows_app_id():
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AIGCDetector.NightCity.Desktop")
    except Exception:
        pass


def main():
    set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("城市夜景AIGC图像检测系统")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    if APP_ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
