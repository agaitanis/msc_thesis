import cubicasa5k.labels as ccl
import numpy as np
import sys
import tensorflow as tf
from contextlib import contextmanager
from distinctipy import distinctipy
from PIL import Image
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


_LABEL_DIVISOR = 256


class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()

        self._img_label = None
        self._tree_view = None
        self._item_model = None
        self._img_fname = None
        self._model = None
        self._output = None

        self._create_win()


    def clear_list(self):
        self._item_model.clear()
        self._item_model.setHorizontalHeaderLabels(("Elements",))


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
        self._img_label.setMinimumSize(QSize(500, 400))

        self._tree_view = QTreeView()
        h_layout.addWidget(self._tree_view)
        self._tree_view.setMinimumWidth(250)
        self._tree_view.setAlternatingRowColors(True)
        self._tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self._item_model = QStandardItemModel()
        self._tree_view.setModel(self._item_model)
        self.clear_list()

        selection_model = self._tree_view.selectionModel()
        selection_model.selectionChanged.connect(self._selection_changed)

        self._predict_button = QPushButton("Detect elements")
        v_layout.addWidget(self._predict_button)
        self._predict_button.clicked.connect(self._detect_elements)
        self._predict_button.setEnabled(False)


    def _open_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open Image")
        file_dialog.setNameFilter("Images (*.png)")

        if file_dialog.exec() == QFileDialog.Accepted:
            self.clear_list()
            self._img_fname = file_dialog.selectedFiles()[0]
            self._img_label.setPixmap(QPixmap.fromImage(QImage(self._img_fname)))
            self._predict_button.setEnabled(True)


    def _selection_changed(self):
        indices = self._tree_view.selectedIndexes()
        for index in indices:
            item = self._item_model.itemFromIndex(index)
            # instance = item.data())
            # instance_mask = np.where(instance_pred == instance, 1, 0)
            # FIXME

    
    def _detect_elements_core(self):
        self.clear_list()

        if self._model is None:
            self._model = tf.saved_model.load("cubicasa5k/model")

        img_array = np.array(Image.open(self._img_fname))

        self._output = self._model(tf.cast(img_array, tf.uint8))
        # self._output is a dict with keys: 
        # center_heatmap, instance_center_pred, instance_pred, 
        # panoptic_pred, offset_map, semantic_pred, 
        # semantic_logits, instance_scores, semantic_probs

        panoptic_pred = self._output["panoptic_pred"].numpy()

        panoptic = np.unique(panoptic_pred)
        panoptic.sort()

        labels = np.unique(panoptic // _LABEL_DIVISOR)
        labels.sort()

        bold_font = QFont()
        bold_font.setBold(True)

        for label in labels:
            label_str = ccl.label_to_str[label]

            parent_str = label_str
            if parent_str != "Background":
                parent_str += "s"
            parent = QStandardItem(parent_str)
            parent.setFont(bold_font)
            parent.setEditable(False)
            self._item_model.appendRow(parent)

            instance_pred = np.where(panoptic_pred // _LABEL_DIVISOR == label, 
                panoptic_pred % _LABEL_DIVISOR, -1)
            instances = np.unique(instance_pred)
            instances.sort()

            for i, instance in enumerate(instances):
                if instance == -1:
                    continue
                if instance == 0:
                    parent.setData(instance)
                    continue

                child_str = f"{label_str} {i}"
                child = QStandardItem(child_str)
                child.setData(instance)
                parent.appendRow(child)


    @contextmanager
    def _detect_elements_context(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._predict_button.setEnabled(False)
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()
            self._predict_button.setEnabled(True)


    def _detect_elements(self):
        with self._detect_elements_context():
            self._detect_elements_core()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget{font-size: 11pt;}")

    win = MainWin()
    win.show()

    app.exec()


if __name__ == '__main__':
    main()