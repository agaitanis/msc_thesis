import sys
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog


class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self._file_dialog = QFileDialog(self)
        self._create_win()


    def _create_win(self):
        self.setWindowTitle("Floor Plan Recognition")
        self.setMinimumSize(QSize(800, 600))

        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open image")
        open_action.triggered.connect(self._open_image)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(open_action)
    

    def _draw_image(self, fname):
        pass # FIXME


    def _open_image(self):
        self._file_dialog = QFileDialog(self)
        self._file_dialog.setWindowTitle("Open Image")
        self._file_dialog.setNameFilter("Images (*.png)")

        if self._file_dialog.exec() == QFileDialog.Accepted:
            fname = self._file_dialog.selectedFiles()[0]
            self._draw_image(fname)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget{font-size: 11pt;}")

    win = MainWin()
    win.show()

    app.exec()


if __name__ == '__main__':
    main()