import cubicasa5k.labels as ccl
import numpy as np
import sys
import tensorflow as tf
from contextlib import contextmanager
from PIL import Image
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, 
    QVBoxLayout, QHBoxLayout, QLabel, QWidget, QTreeView, QPushButton)


@contextmanager
def wait_cursor():
    try:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        yield
    finally:
        QApplication.restoreOverrideCursor()


class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self._img_label = None
        self._tree_view = None
        self._img_fname = None
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

        self._img_label = QLabel()
        h_layout.addWidget(self._img_label)
        self._img_label.setMinimumSize(QSize(600, 500))

        self._tree_view = QTreeView()
        h_layout.addWidget(self._tree_view)
        self._tree_view.setMinimumWidth(200)

        self._predict_button = QPushButton("Predict")
        v_layout.addWidget(self._predict_button)
        self._predict_button.clicked.connect(self._predict)
        self._predict_button.setEnabled(False)


    def _open_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open Image")
        file_dialog.setNameFilter("Images (*.png)")

        if file_dialog.exec() == QFileDialog.Accepted:
            self._img_fname = file_dialog.selectedFiles()[0]
            self._img_label.setPixmap(QPixmap.fromImage(QImage(self._img_fname)))
            self._predict_button.setEnabled(True)
    

    def _predict(self):
        with wait_cursor():
            model = tf.saved_model.load("cubicasa5k/model")
            img_array = np.array(Image.open(self._img_fname))
            output = model(tf.cast(img_array, tf.uint8))
            # output is a dict with keys: 
            # center_heatmap, instance_center_pred, instance_pred, 
            # panoptic_pred, offset_map, semantic_pred, 
            # semantic_logits, instance_scores, semantic_probs

            panoptic_pred = output['panoptic_pred']
            semantic_pred = output['semantic_pred']
            instance_pred = output['instance_pred']
            print(tf.shape(instance_pred))
            panoptic_pred = panoptic_pred.numpy()
            semantic_pred = semantic_pred.numpy()
            instance_pred = instance_pred.numpy()
            print(instance_pred.shape)

            print("panoptic_pred =", np.unique(panoptic_pred))
            print("semantic_pred =", np.unique(semantic_pred))
            print("instance_pred =", np.unique(instance_pred))

            im = Image.fromarray(ccl.get_colormap()[semantic_pred[0]])
            im.save("cubicasa5k/semantice_pred.png")


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget{font-size: 11pt;}")

    win = MainWin()
    win.show()

    app.exec()


if __name__ == '__main__':
    main()