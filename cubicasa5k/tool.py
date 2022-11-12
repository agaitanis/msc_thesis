import cubicasa5k.labels as ccl
import numpy as np
import random
import os
import sys
import tensorflow as tf
from contextlib import contextmanager
from distinctipy import distinctipy
from math import sqrt
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtPrintSupport import *
from PyQt6.QtWidgets import *
from PIL import Image, ImageQt


_LABEL_DIVISOR = 256


class PanopticIdData():
    def __init__(self, color=None, center=None, is_selected=False):
        self.color = color
        self.center = center
        self.is_selected = is_selected

    
class ImgLabel(QLabel):
    def __init__(self, panoptic_id_to_data, edges):
        super().__init__()
        self._panoptic_id_to_data = panoptic_id_to_data
        self._edges = edges
        self._scale_factor = 1.0


    def get_scale_factor(self):
        return self._scale_factor


    def set_scale_factor(self, val):
        self._scale_factor = val
        if val == 1.0:
            self.adjustSize()
        else:
            self.resize(val * self.pixmap().size())


    def paintEvent(self, event):
        super().paintEvent(event)

        r = 10.0

        painter = QPainter(self)

        for _, data in self._panoptic_id_to_data.items():
            color = data.color
            center = data.center

            if center is None:
                continue

            is_selected = data.is_selected

            x = center[1]*self._scale_factor
            y = center[0]*self._scale_factor
            point = QPointF(x, y)
            width = 2
            alpha = 150

            if is_selected:
                width = 5
                alpha = 255

            painter.setPen(QPen(QColor(0, 0, 0, alpha), width))
            painter.setBrush(QBrush(QColor(color[0], color[1], color[2], alpha)))
            painter.drawEllipse(point, r, r)
        
        alpha = 150
        width = 2
        
        for id1, id2 in self._edges:
            center1 = self._panoptic_id_to_data[id1].center
            center2 = self._panoptic_id_to_data[id2].center
            x1 = center1[1]*self._scale_factor
            y1 = center1[0]*self._scale_factor
            point1 = QPointF(x1, y1)
            x2 = center2[1]*self._scale_factor
            y2 = center2[0]*self._scale_factor
            point2 = QPointF(x2, y2)
            vec = (point2 - point1)/QLineF(point1, point2).length()
            point1 += r*vec
            point2 -= r*vec

            painter.setPen(QPen(QColor(0, 0, 0, alpha), width))
            painter.drawLine(point1, point2)



class ItemModel(QStandardItemModel):
    def __init__(self, panoptic_id_to_data):
        super().__init__()
        self._panoptic_id_to_data = panoptic_id_to_data
    

    def data(self, index, role):
        if role == Qt.ItemDataRole.DecorationRole and len(self._panoptic_id_to_data) > 0:
            item = self.itemFromIndex(index)
            panoptic_id = item.data()
            if panoptic_id is not None:
                color = self._panoptic_id_to_data[panoptic_id].color
                return QColor(color[0], color[1], color[2])

        return super().data(index, role)
    

class Icon(QIcon):
    def __init__(self, filename):
        filename = os.path.join(os.path.dirname(__file__), "icons", filename)
        super().__init__(filename)


class Separator(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)


def _get_pixel_neibs(img_array, i, j):
    neibs = []

    if i-1 >= 0:
        neibs.append(img_array[i-1, j])
    if i+1 < img_array.shape[0]:
        neibs.append(img_array[i+1, j])

    if j-1 >= 0:
        neibs.append(img_array[i, j-1])
    if j+1 < img_array.shape[1]:
        neibs.append(img_array[i, j+1])

    return neibs


def _euclidean_dist(p1, p2):
    x = p1[0] - p2[0]
    y = p1[1] - p2[1]
    return sqrt(x*x + y*y)


class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()

        self._img_label = None
        self._img_qt = None
        self._scroll_area = None
        self._img_file_name = None
        self._tree_view = None
        self._detect_elements_button = None
        self._create_graph_button = None
        self._model = None
        self._panoptic_id_to_data = {}
        self._graph = {}
        self._edges = {}

        self._create_win()


    def _create_win(self):
        self.setWindowTitle("Floor Plan Recognition")
        self.showMaximized()

        menu_bar = self.menuBar()

        new_action = QAction(Icon("new_file.svg"), "New", self, shortcut="Ctrl+N", 
            triggered=self._new_file)
        open_action = QAction(Icon("open_file.svg"), "Open...", self, shortcut="Ctrl+O", 
            triggered=self._open_file)
        exit_action = QAction("Exit", self, shortcut="Ctrl+Q", triggered=self.close)

        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(new_action)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        fit_to_window_action = QAction(Icon("fit_to_window.svg"), "Fit to window", self, 
            shortcut="Ctrl+0", triggered=self._fit_to_window)
        initial_size_action = QAction(Icon("original_size.svg"), "Original size", self, 
            shortcut="Ctrl+1", triggered=self._original_size)
        zoom_in_action = QAction(Icon("zoom_in.svg"), "Zoom in", self, 
            shortcut="Ctrl++", triggered=self._zoom_in)
        zoom_out_action = QAction(Icon("zoom_out.svg"), "Zoom out", self, 
            shortcut="Ctrl+-", triggered=self._zoom_out)

        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(fit_to_window_action)
        view_menu.addAction(initial_size_action)
        view_menu.addAction(zoom_in_action)
        view_menu.addAction(zoom_out_action)

        v_layout = QVBoxLayout()

        widget = QWidget()
        widget.setLayout(v_layout)
        self.setCentralWidget(widget)

        h_layout = QHBoxLayout()
        h_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        v_layout.addLayout(h_layout)

        button = QPushButton()
        button.setIcon(Icon("new_file.svg"))
        button.clicked.connect(self._new_file)
        button.setToolTip("New")
        h_layout.addWidget(button)

        button = QPushButton()
        button.setIcon(Icon("open_file.svg"))
        button.clicked.connect(self._open_file)
        button.setToolTip("Open")
        h_layout.addWidget(button)

        h_layout.addWidget(Separator())
        
        button = QPushButton()
        button.setIcon(Icon("fit_to_window.svg"))
        button.clicked.connect(self._fit_to_window)
        button.setToolTip("Fit to window")
        h_layout.addWidget(button)

        button = QPushButton()
        button.setIcon(Icon("original_size.svg"))
        button.clicked.connect(self._original_size)
        button.setToolTip("Original size")
        h_layout.addWidget(button)

        button = QPushButton()
        button.setIcon(Icon("zoom_in.svg"))
        button.clicked.connect(self._zoom_in)
        button.setToolTip("Zoom in")
        h_layout.addWidget(button)
        h_layout.addWidget(button)

        button = QPushButton()
        button.setIcon(Icon("zoom_out.svg"))
        button.clicked.connect(self._zoom_out)
        button.setToolTip("Zoom out")
        h_layout.addWidget(button)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        v_layout.addWidget(splitter)

        self._tree_view = QTreeView()
        self._tree_view.setAlternatingRowColors(True)
        self._tree_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._tree_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        splitter.addWidget(self._tree_view)

        self._item_model = ItemModel(self._panoptic_id_to_data)
        self._tree_view.setModel(self._item_model)
        self._clear_list()

        selection_model = self._tree_view.selectionModel()
        selection_model.selectionChanged.connect(self._selection_changed)

        self._scroll_area = QScrollArea()
        self._scroll_area.setBackgroundRole(QPalette.ColorRole.Dark)
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll_area.setVisible(True)

        self._img_label = ImgLabel(self._panoptic_id_to_data, self._edges)
        self._img_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._img_label.setScaledContents(True)
        self._img_label.resize(300, 300)
        self._scroll_area.setWidget(self._img_label)
        splitter.addWidget(self._scroll_area)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        h_layout = QHBoxLayout()
        v_layout.addLayout(h_layout)

        self._detect_elements_button = QPushButton("Detect elements")
        self._detect_elements_button.clicked.connect(self._detect_elements)
        self._detect_elements_button.setEnabled(False)
        h_layout.addWidget(self._detect_elements_button)

        self._create_graph_button = QPushButton("Create graph")
        self._create_graph_button.clicked.connect(self._create_graph)
        self._create_graph_button.setEnabled(False)
        h_layout.addWidget(self._create_graph_button)


    def _load_img(self, img_qt):
        self._img_qt = img_qt
        self._img_label.setPixmap(QPixmap.fromImage(img_qt))

    
    def _adjust_scroll_bar(self, scroll_bar, factor):
        scroll_bar.setValue(int(factor * scroll_bar.value() + ((factor - 1) * scroll_bar.pageStep() / 2)))


    def _scale_img(self, factor):
        if self._img_qt is None:
            return

        self._load_img(self._img_qt)
        self._img_label.set_scale_factor(factor*self._img_label.get_scale_factor())

        self._adjust_scroll_bar(self._scroll_area.horizontalScrollBar(), factor)
        self._adjust_scroll_bar(self._scroll_area.verticalScrollBar(), factor)


    def _zoom_in(self):
        self._scale_img(1.25)


    def _zoom_out(self):
        self._scale_img(0.8)


    def _original_size(self):
        self._img_label.set_scale_factor(1.0)

    
    def _fit_to_window(self):
        src_size = self._img_label.size()
        trg_size = self._scroll_area.size()

        src_w = src_size.width()
        src_h = src_size.height()
        trg_w = trg_size.width() - self._scroll_area.verticalScrollBar().size().width()
        trg_h = trg_size.height() - self._scroll_area.horizontalScrollBar().size().height()

        if trg_w/trg_h < src_w/src_h:
            self._scale_img(trg_w/src_w)
        else:
            self._scale_img(trg_h/src_h)


    def _new_file(self):
        self._clear_list()
        self._img_label.clear()
        self._img_qt = None
        self._img_file_name = None
        self._detect_elements_button.setEnabled(False)
        self._create_graph_button.setEnabled(False)
        self._panoptic_id_to_data.clear()
        self._graph.clear()


    def _open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
            "Images (*.png *.jpeg *.jpg *.bmp *.gif)")
        if not filename:
            return

        image = QImage(filename)
        if image.isNull():
            QMessageBox.information(self, "Image Viewer", "Cannot load %s." % filename)
            return

        self._new_file()
        
        self._load_img(image)

        self._img_file_name = filename
        self._img_label.set_scale_factor(1.0)
        self._fit_to_window()
        self._detect_elements_button.setEnabled(True)
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

        for panoptic_id in self._panoptic_id_to_data.keys():
            self._panoptic_id_to_data[panoptic_id].is_selected = False
            
        if len(items) == 0:
            self._load_img(ImageQt.ImageQt(background))
            return

        panoptic_pred = self._output["panoptic_pred"].numpy()[0][:, :, np.newaxis]
        foreground_array = np.zeros((panoptic_pred.shape[0], panoptic_pred.shape[1], 4)).astype(np.uint8)

        for item in items:
            panoptic_id = item.data()
            if panoptic_id is None:
                continue

            self._panoptic_id_to_data[panoptic_id].is_selected = True

            color = self._panoptic_id_to_data[panoptic_id].color
            foreground_array += np.where(panoptic_pred == panoptic_id,
                (color[0], color[1], color[2], 200), (0, 0, 0, 0)).astype(np.uint8)

        foreground = Image.fromarray(foreground_array)
        background.paste(foreground, (0, 0), foreground)
        self._load_img(ImageQt.ImageQt(background))
    

    def _detect_elements_core(self):
        self._clear_list()

        if self._model is None:
            self._model = tf.saved_model.load("cubicasa5k/model")

        img_array = np.array(Image.open(self._img_file_name).convert("RGB"))

        self._output = self._model(tf.cast(img_array, tf.uint8))
        # self._output is a dict with keys: 
        # center_heatmap, instance_center_pred, instance_pred, 
        # panoptic_pred, offset_map, semantic_pred, 
        # semantic_logits, instance_scores, semantic_probs

        panoptic_pred = self._output["panoptic_pred"].numpy()[0]

        panoptic_ids = np.unique(panoptic_pred)
        panoptic_ids.sort()

        labels = np.unique(panoptic_ids // _LABEL_DIVISOR)
        labels.sort()

        for label in labels:
            if label == ccl.Label.BACKGROUND:
                continue

            label_str = ccl.label_to_str[label]
            parent_str = label_str + "s"
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
                child.setEditable(False)
                child.setData(panoptic_id)
                parent.appendRow(child)
        
        self._tree_view.expandAll()
        
        random.seed(0)
        colors = (np.array(distinctipy.get_colors(len(panoptic_ids)))*255).astype(np.uint8)

        for panoptic_id, color in zip(panoptic_ids, colors):
            self._panoptic_id_to_data[panoptic_id] = PanopticIdData(color)

        self._create_graph_button.setEnabled(True)


    @contextmanager
    def _wait_cursor(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()


    def _detect_elements(self):
        with self._wait_cursor():
            self._detect_elements_core()


    def _calc_edge_weight(self, id1, id2):
        center1 = self._panoptic_id_to_data[id1].center
        center2 = self._panoptic_id_to_data[id2].center

        return _euclidean_dist(center1, center2)
    

    def _create_graph_core(self):
        panoptic_pred = self._output["panoptic_pred"].numpy()[0]
        instance_center_pred = self._output["instance_center_pred"].numpy()[0]

        panoptic_id_to_center_pred_max = {}
        graph = {}

        for i in range(panoptic_pred.shape[0]):
            for j in range(panoptic_pred.shape[1]):
                id = panoptic_pred[i][j]
                label = id // _LABEL_DIVISOR
                if label in (ccl.Label.BACKGROUND, ccl.Label.WALL):
                    continue

                if instance_center_pred[i][j] > panoptic_id_to_center_pred_max.get(id, -1.0):
                    panoptic_id_to_center_pred_max[id] = instance_center_pred[i][j]
                    self._panoptic_id_to_data[id].center = (i, j)

                neibs = _get_pixel_neibs(panoptic_pred, i, j)

                for neib in neibs:
                    if neib == id:
                        continue
                    neib_label = neib // _LABEL_DIVISOR
                    if neib_label in (ccl.Label.BACKGROUND, ccl.Label.WALL):
                        continue
                    if (id, neib) not in graph:
                        graph[(id, neib)] = 1.0
        
        for (id, neib), _ in graph.items():
            w = self._calc_edge_weight(id, neib)
            self._graph[(id, neib)] = w
            self._edges[(min(id, neib), max(id, neib))] = w
        
        self._img_label.update()

    
    def _create_graph(self):
        with self._wait_cursor():
            self._create_graph_core()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    win = MainWin()
    win.show()

    sys.exit(app.exec())