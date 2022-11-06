import cubicasa5k.labels as ccl
import numpy as np
import random
import os
import sys
import tensorflow as tf
from contextlib import contextmanager
from distinctipy import distinctipy
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtPrintSupport import *
from PyQt6.QtWidgets import *
from PIL import Image, ImageQt


_LABEL_DIVISOR = 256


class ItemModel(QStandardItemModel):
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


class Icon(QIcon):
    def __init__(self, filename):
        filename = os.path.join(os.path.dirname(__file__), "icons", filename)
        super().__init__(filename)


class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()

        self._scale_factor = 0.0
        self._img_label = None
        self._scroll_area = None
        self._img_file_name = None
        self._tree_view = None
        self._model = None

        self._create_win()


    def _create_win(self):
        self.setWindowTitle("Floor Plan Recognition")
        self.resize(800, 500)

        menu_bar = self.menuBar()

        open_action = QAction(Icon("open_file.svg"), "&Open...", self, shortcut="Ctrl+O", 
            triggered=self._open_image)
        exit_action = QAction("&Exit", self, shortcut="Ctrl+Q", triggered=self.close)

        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        zoom_in_action = QAction(Icon("zoom_in.svg"), "Zoom &In (25%)", self, 
            shortcut="Ctrl++", triggered=self._zoom_in)
        zoom_out_action = QAction(Icon("zoom_out.svg"), "Zoom &Out (25%)", self, 
            shortcut="Ctrl+-", triggered=self._zoom_out)
        initial_size_action = QAction(Icon("original_size.svg"), "Original &Size", self, 
            shortcut="Ctrl+1", triggered=self._original_size)

        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(zoom_in_action)
        view_menu.addAction(zoom_out_action)
        view_menu.addAction(initial_size_action)

        v_layout = QVBoxLayout()
        v_layout.setContentsMargins(0, 0, 0, 0)

        widget = QWidget()
        widget.setLayout(v_layout)
        self.setCentralWidget(widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        v_layout.addWidget(splitter)

        self._tree_view = QTreeView()
        splitter.addWidget(self._tree_view)
        self._tree_view.setAlternatingRowColors(True)
        self._tree_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self._item_model = ItemModel()
        self._tree_view.setModel(self._item_model)
        self._clear_list()

        selection_model = self._tree_view.selectionModel()
        selection_model.selectionChanged.connect(self._selection_changed)

        self._img_label = QLabel()
        self._img_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._img_label.setScaledContents(True)
        self._img_label.resize(300, 300)

        self._scroll_area = QScrollArea()
        splitter.addWidget(self._scroll_area)
        self._scroll_area.setBackgroundRole(QPalette.ColorRole.Dark)
        self._scroll_area.setWidget(self._img_label)
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._predict_button = QPushButton("Detect elements")
        v_layout.addWidget(self._predict_button)
        self._predict_button.clicked.connect(self._detect_elements)
        self._predict_button.setEnabled(False)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)


    def _load_img(self, img_qt):
        self._img_qt = img_qt
        self._img_label.setPixmap(QPixmap.fromImage(img_qt))


    def _adjust_scroll_bar(self, scroll_bar, factor):
        scroll_bar.setValue(int(factor * scroll_bar.value() + ((factor - 1) * scroll_bar.pageStep() / 2)))


    def _scale_img(self, factor):
        self._scale_factor *= factor
        self._load_img(self._img_qt)
        self._img_label.resize(self._scale_factor * self._img_label.pixmap().size())

        self._adjust_scroll_bar(self._scroll_area.horizontalScrollBar(), factor)
        self._adjust_scroll_bar(self._scroll_area.verticalScrollBar(), factor)


    def _zoom_in(self):
        self._scale_img(1.25)


    def _zoom_out(self):
        self._scale_img(0.8)


    def _original_size(self):
        self._img_label.adjustSize()
        self._scale_factor = 1.0


    def _open_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
            "Images (*.png *.jpeg *.jpg *.bmp *.gif)")
        if not filename:
            return

        image = QImage(filename)
        if image.isNull():
            QMessageBox.information(self, "Image Viewer", "Cannot load %s." % filename)
            return
        
        self._load_img(image)

        self._img_file_name = filename
        self._scale_factor = 1.0
        self._scroll_area.setVisible(True)
        self._img_label.adjustSize()
        self._predict_button.setEnabled(True)
        self._clear_list()
    

    def _clear_list(self):
        self._item_model.clear()
        self._item_model.setHorizontalHeaderLabels(("Elements",))

    
    def _get_selected_childless_items(self):
        indexes = set()
        selected_indexes = self._tree_view.selectedIndexes()

        for index in selected_indexes:
            item = self._item_model.itemFromIndex(index)
            child_items_num = item.rowCount()
            
            if child_items_num > 0:
                for i in range(child_items_num):
                    indexes.add(item.child(i).index())
            else:
                indexes.add(index)
        
        items = [self._item_model.itemFromIndex(x) for x in indexes]
        
        return items


    def _selection_changed(self):
        background = Image.open(self._img_file_name)

        items = self._get_selected_childless_items()

        if len(items) == 0:
            self._load_img(ImageQt.ImageQt(background))
            return

        panoptic_pred = self._output["panoptic_pred"].numpy()[0][:, :, np.newaxis]
        foreground_array = np.zeros((panoptic_pred.shape[0], panoptic_pred.shape[1], 4)).astype(np.uint8)

        for item in items:
            panoptic_id = item.data()
            if panoptic_id is None:
                continue

            color = self._panoptic_id_to_color[panoptic_id]
            foreground_array += np.where(panoptic_pred == panoptic_id,
                (color[0], color[1], color[2], 200), (0, 0, 0, 0)).astype(np.uint8)

        foreground = Image.fromarray(foreground_array)
        background.paste(foreground, (0, 0), foreground)
        self._load_img(ImageQt.ImageQt(background))
    

    def _detect_elements_core(self):
        self._clear_list()

        if self._model is None:
            self._model = tf.saved_model.load("cubicasa5k/model")

        img_array = np.array(Image.open(self._img_file_name))

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

        for label in labels:
            label_str = ccl.label_to_str[label]

            parent_str = label_str
            if parent_str != "Background":
                parent_str += "s"
            parent = QStandardItem(parent_str)
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
        
        random.seed(0)
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


if __name__ == '__main__':
    app = QApplication(sys.argv)

    win = MainWin()
    win.show()

    sys.exit(app.exec())