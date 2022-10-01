import sys
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, 
    QVBoxLayout, QHBoxLayout, QLabel, QWidget, QTreeView)


class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self._image_label = None
        self._tree_view = None
        self._create_win()


    def _create_win(self):
        self.setWindowTitle("Floor Plan Recognition")
        # self.setMinimumSize(QSize(800, 600))

        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open image")
        open_action.triggered.connect(self._open_image)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(open_action)

        v_layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(v_layout)
        self.setCentralWidget(widget)

        h_layout = QHBoxLayout()
        v_layout.addLayout(h_layout)
        h_layout.setAlignment(Qt.AlignCenter)

        self._image_label = QLabel()
        h_layout.addWidget(self._image_label)
        self._image_label.setMinimumSize(QSize(600, 500))

        self._tree_view = QTreeView()
        h_layout.addWidget(self._tree_view)
        self._tree_view.setMinimumWidth(200)
    

    def _draw_image(self, fname):
        image = QImage(fname)
        self._image_label.setPixmap(QPixmap.fromImage(image))


    def _open_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open Image")
        file_dialog.setNameFilter("Images (*.png)")

        if file_dialog.exec() == QFileDialog.Accepted:
            fname = file_dialog.selectedFiles()[0]
            self._draw_image(fname)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget{font-size: 11pt;}")

    win = MainWin()
    win.show()

    app.exec()


if __name__ == '__main__':
    main()