import cubicasa5k.labels as ccl
import numpy as np
import sys
import tensorflow as tf
from contextlib import contextmanager
from distinctipy import distinctipy
from PIL import Image, ImageQt
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *


_LABEL_DIVISOR = 256


class MyItemModel(QStandardItemModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.panoptic_id_to_color = None


    def data(self, index, role):
        if role == Qt.ItemDataRole.DecorationRole and self.panoptic_id_to_color is not None:
            item = self.itemFromIndex(index)
            panoptic_id = item.data()
            if panoptic_id is not None:
                color = self.panoptic_id_to_color[panoptic_id]
                return QColor(color[0], color[1], color[2])
        return super().data(index, role)
    

    def clear(self):
        super().clear()
        self.panoptic_id_to_color = None


class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()

        self._img_label = None
        self._tree_view = None
        self._item_model = None
        self._img_fname = None
        self._model = None
        self._output = None
        self._panoptic_id_to_color = None

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

        self._img_label = QLabel()
        h_layout.addWidget(self._img_label)
        self._img_label.setMinimumSize(QSize(500, 400))

        self._tree_view = QTreeView()
        h_layout.addWidget(self._tree_view)
        self._tree_view.setMinimumWidth(250)
        self._tree_view.setAlternatingRowColors(True)
        self._tree_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self._item_model = MyItemModel()
        self._tree_view.setModel(self._item_model)
        self.clear_list()

        selection_model = self._tree_view.selectionModel()
        selection_model.selectionChanged.connect(self._selection_changed)

        self._predict_button = QPushButton("Detect elements")
        v_layout.addWidget(self._predict_button)
        self._predict_button.clicked.connect(self._detect_elements)
        self._predict_button.setEnabled(False)


    def _open_image(self):
        filename = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png)", None)

        if len(filename[0]) > 0:
            self.clear_list()
            self._img_fname = filename[0]
            self._img_label.setPixmap(QPixmap.fromImage(QImage(self._img_fname)))
            self._predict_button.setEnabled(True)


    def _selection_changed(self):
        indices = self._tree_view.selectedIndexes()
        if len(indices) == 0:
            return

        panoptic_pred = self._output["panoptic_pred"].numpy()[0]

        for index in indices:
            item = self._item_model.itemFromIndex(index)
            panoptic_id = item.data()

            if panoptic_id is not None:
                color = self._panoptic_id_to_color[panoptic_id]
                img = np.where(panoptic_pred[:, :, np.newaxis] == panoptic_id, 
                    (color[0], color[1], color[2], 200), (0, 0, 0, 0)).astype(np.uint8)
                im1 = Image.fromarray(img)
                im2 = Image.open(self._img_fname)
                im2.paste(im1, (0, 0), im1)
                qim = ImageQt.ImageQt(im2)
                self._img_label.setPixmap(QPixmap.fromImage(qim))

    
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

        panoptic_ids = np.unique(panoptic_pred)
        panoptic_ids.sort()

        labels = np.unique(panoptic_ids // _LABEL_DIVISOR)
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

                panoptic_id = label*_LABEL_DIVISOR + instance

                if instance == 0:
                    parent.setData(panoptic_id)
                    continue

                child_str = f"{label_str} {i}"
                child = QStandardItem(child_str)
                child.setData(panoptic_id)
                parent.appendRow(child)
        
        self._tree_view.expandAll()
        
        colors = (np.array(distinctipy.get_colors(len(panoptic_ids)))*255).astype(np.uint8)
        self._panoptic_id_to_color = dict(zip(panoptic_ids, colors))
        self._item_model.panoptic_id_to_color = self._panoptic_id_to_color


    @contextmanager
    def _detect_elements_context(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
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