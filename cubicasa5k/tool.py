from __future__ import annotations
import cubicasa5k.labels as ccl
import functools
import numpy as np
import os
import queue
import sys
import tensorflow as tf
from contextlib import contextmanager
from distinctipy import distinctipy
from enum import IntEnum
from collections import defaultdict
from math import sqrt
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtPrintSupport import *
from PyQt6.QtWidgets import *
from PIL import Image, ImageQt
from xml.dom import minidom


_LABEL_DIVISOR = 256
_NODE_RADIUS = 12
_SELECTED_COLOR = (0, 0, 100)
_PATH_COLOR = (0, 136, 190)


class _ItemType(IntEnum):
    WALL = 0
    ROOM = 1
    DOOR = 2
    NODE = 3


class _Mark(IntEnum):
    NONE = 0
    EXIT = 1


def _label_to_item_type(label):
    if label == ccl.Label.WALL:
        return _ItemType.WALL
    elif label == ccl.Label.ROOM:
        return _ItemType.ROOM
    elif label == ccl.Label.DOOR:
        return _ItemType.DOOR
    else:
        raise ValueError(f"Unknown label: {label}")


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


def _dist_to_edge_for_pick(p, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    x, y = p
    if (x1 - x)*(x2 - x) + (y1 - y)*(y2 - y) > 0:
        return None
    dist = abs((x2 - x1)*(y1 - y) - (x1 - x)*(y2 - y1)) / sqrt((x2-x1)**2 + (y2-y1)**2)
    if dist <= 12:
        return dist
    return None


@contextmanager
def _wait_cursor():
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


class _Elem():
    def __init__(self, color, item:QStandardItem):
        self.color = color
        self.item = item
        self.is_selected = False

        
class _Node():
    def __init__(self, color, item:QStandardItem, center=None):
        self.color = color
        self.item = item
        self.center = center
        self.is_selected = False
        self.highlight_for_path = False
        self.mark = _Mark.NONE
        self.path = []


class _EdgeData():
    def __init__(self):
        self.is_selected = False
        self.highlight_for_path = False


class _ScrollArea(QScrollArea):
    def __init__(self, win: _MainWin):
        super().__init__()
        self._win = win


    def wheelEvent(self, event: QWheelEvent):
        speed = event.angleDelta().y()
        if speed > 0:
            self._win.scale_img(1.1, event.position())
        elif speed < 0:
            self._win.scale_img(0.9, event.position())

    
class _ImgLabel(QLabel):
    def __init__(self, win: _MainWin):
        super().__init__()
        self._win = win
        self._move_img_is_allowed = False
        self._move_node_is_allowed = False
        self._start_pos = None
        self._picked_node_id = None
    

    def _pick_node(self, pos):
        nearest_node_id = None
        nearest_node = None
        min_dist = sys.float_info.max

        for id, node in self._win.id_to_node.items():
            x1 = node.center[0]*self._win.scale_factor
            y1 = node.center[1]*self._win.scale_factor
            x2 = pos.x()
            y2 = pos.y()
            dist = _euclidean_dist((x1, y1), (x2, y2))
            if dist < min_dist:
                min_dist = dist
                nearest_node_id = id
                nearest_node = node

        if min_dist <= _NODE_RADIUS + 2:
            if QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier:
                nearest_node.is_selected = not nearest_node.is_selected
            elif QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier:
                pass
            else:
                nearest_node.is_selected = True

            if nearest_node.is_selected:
                self._win.tree_view.selectionModel().select(nearest_node.item.index(),
                    QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
            else:
                self._win.tree_view.selectionModel().select(nearest_node.item.index(),
                    QItemSelectionModel.SelectionFlag.Deselect | QItemSelectionModel.SelectionFlag.Rows)

            self._win.redraw()

            return nearest_node_id

        return None
    

    def _pick_edge(self, pos):
        nearest_edge = None
        nearest_edge_data = None
        min_dist = sys.float_info.max

        for edge, data in self._win.edges.items():
            node1 = self._win.id_to_node[edge[0]]
            node2 = self._win.id_to_node[edge[1]]
            x1 = node1.center[0]*self._win.scale_factor
            y1 = node1.center[1]*self._win.scale_factor
            x2 = node2.center[0]*self._win.scale_factor
            y2 = node2.center[1]*self._win.scale_factor

            dist = _dist_to_edge_for_pick((pos.x(), pos.y()), (x1, y1), (x2, y2))

            if dist is not None and dist < min_dist:
                min_dist = dist
                nearest_edge = edge
                nearest_edge_data = data
        
        if nearest_edge_data is not None:
            if QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier:
                nearest_edge_data.is_selected = not nearest_edge_data.is_selected
            elif QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier:
                pass
            else:
                nearest_edge_data.is_selected = True

            self._win.redraw()

            return nearest_edge

        return None
    

    def _deselect_rest_of_items_if_needed(self, picked_node_id, picked_edge):
        if QApplication.keyboardModifiers() in (Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.ShiftModifier):
            return

        for id, node in self._win.id_to_node.items():
            if id != picked_node_id:
                node.is_selected = False
                self._win.tree_view.selectionModel().select(node.item.index(), 
                    QItemSelectionModel.SelectionFlag.Deselect | QItemSelectionModel.SelectionFlag.Rows)
        for edge, data in self._win.edges.items():
            if edge != picked_edge:
                data.is_selected = False

        self._win.redraw()


    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton and self._win.find_path_button.isEnabled():
            menu = QMenu(self)
            menu.addAction("Add node here", functools.partial(self._win.add_node_at_pos, event.pos()))
            menu.exec(QCursor.pos())
            return

        self._picked_node_id = self._pick_node(event.pos())
        picked_edge = None
        if self._picked_node_id is None:
            picked_edge = self._pick_edge(event.pos())
        self._deselect_rest_of_items_if_needed(self._picked_node_id, picked_edge)

        self._move_node_is_allowed = self._picked_node_id is not None and\
            QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier
        
        self._move_img_is_allowed = QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier and\
            (self._win.scroll_area.horizontalScrollBar().isVisible() or\
            self._win.scroll_area.verticalScrollBar().isVisible())
        
        if self._move_img_is_allowed:
            self._start_pos = event.pos()

        if self._move_node_is_allowed or self._move_img_is_allowed:
            QApplication.setOverrideCursor(Qt.CursorShape.ClosedHandCursor)
    

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._move_img_is_allowed or self._move_node_is_allowed:
            QApplication.restoreOverrideCursor()

        self._move_node_is_allowed = False
        self._move_img_is_allowed = False
    

    def _move_node(self, pos: QPoint):
        pos.setX(max(0, min(pos.x(), self.width())))
        pos.setY(max(0, min(pos.y(), self.height())))

        node = self._win.id_to_node[self._picked_node_id]
        node.center = (pos.x()/self._win.scale_factor, pos.y()/self._win.scale_factor)

        for edge in self._win.graph.keys():
            if edge[0] == self._picked_node_id:
                self._win.graph[edge] = _euclidean_dist(node.center, self._win.id_to_node[edge[1]].center)
            elif edge[1] == self._picked_node_id:
                self._win.graph[edge] = _euclidean_dist(node.center, self._win.id_to_node[edge[0]].center)

        self._win.clear_paths()
        self._win.recalc_draw()
        self._win.redraw()

    
    def _move_img(self, pos):
        scroll_bar_pos = QPoint(self._win.scroll_area.horizontalScrollBar().value(),
            self._win.scroll_area.verticalScrollBar().value())

        delta = pos - self._start_pos

        new_scroll_bar_pos = scroll_bar_pos - delta

        if new_scroll_bar_pos.x() < self._win.scroll_area.horizontalScrollBar().minimum() or\
            new_scroll_bar_pos.x() > self._win.scroll_area.horizontalScrollBar().maximum():
            delta.setX(0)

        if new_scroll_bar_pos.y() < self._win.scroll_area.verticalScrollBar().minimum() or\
            new_scroll_bar_pos.y() > self._win.scroll_area.verticalScrollBar().maximum():
            delta.setY(0)

        self.move(self.pos() + delta)

        self._win.scroll_area.horizontalScrollBar().setValue(scroll_bar_pos.x() - delta.x())
        self._win.scroll_area.verticalScrollBar().setValue(scroll_bar_pos.y() - delta.y())


    def mouseMoveEvent(self, event: QMouseEvent):
        if self._move_node_is_allowed:
            self._move_node(event.pos())
        elif self._move_img_is_allowed:
            self._move_img(event.pos())
    
       
    def _get_width_alpha(self, is_selected, highlight_for_path):
        if is_selected or highlight_for_path:
            return 6, 255
        else:
            return 2, 150
    

    def _get_pen_color(self, is_selected, highlight_for_path):
        if is_selected:
            return _SELECTED_COLOR
        elif highlight_for_path:
            return _PATH_COLOR
        else:
            return _SELECTED_COLOR
    
    
    def _draw_nodes(self, painter: QPainter):
        r = _NODE_RADIUS

        for _, node in self._win.id_to_node.items():
            x = node.center[0]*self._win.scale_factor
            y = node.center[1]*self._win.scale_factor
            point = QPointF(x, y)

            width, alpha = self._get_width_alpha(node.is_selected, node.highlight_for_path)
            color = self._get_pen_color(node.is_selected, node.highlight_for_path)

            painter.setPen(QPen(QColor(color[0], color[1], color[2], alpha), width))
            painter.setBrush(QBrush(QColor(node.color[0], node.color[1], node.color[2], alpha)))
            painter.drawEllipse(point, r, r)

            if node.mark == _Mark.EXIT:
                color = (np.array(distinctipy.get_text_color(node.color/255))*255).astype(np.uint8)
                painter.setPen(QPen(QColor(color[0], color[1], color[2], alpha), 1))
                painter.setBrush(QBrush(QColor(color[0], color[1], color[2], alpha)))

                painter.drawPolygon(point + QPointF(r*0.5, 0), point + QPointF(-r*0.45, -r*0.45),
                    point + QPointF(-r*0.45, r*0.45))


    def _draw_edges(self, painter: QPainter):
        r = _NODE_RADIUS

        for edge, data in self._win._edges.items():
            center1 = self._win.id_to_node[edge[0]].center
            center2 = self._win.id_to_node[edge[1]].center
            x1 = center1[0]*self._win.scale_factor
            y1 = center1[1]*self._win.scale_factor
            point1 = QPointF(x1, y1)
            x2 = center2[0]*self._win.scale_factor
            y2 = center2[1]*self._win.scale_factor
            point2 = QPointF(x2, y2)
            vec = (point2 - point1)/QLineF(point1, point2).length()
            point1 += r*vec
            point2 -= r*vec

            width, alpha = self._get_width_alpha(data.is_selected, data.highlight_for_path)
            color = self._get_pen_color(data.is_selected, data.highlight_for_path)

            painter.setPen(QPen(QColor(color[0], color[1], color[2], alpha), width))
            painter.drawLine(point1, point2)


    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)

        self._draw_nodes(painter)
        self._draw_edges(painter)

        
class _ItemModel(QStandardItemModel):
    def __init__(self, win: _MainWin):
        super().__init__()
        self._win = win
    

    def data(self, index, role):
        if role != Qt.ItemDataRole.DecorationRole:
            return super().data(index, role)

        item = self.itemFromIndex(index)
        if item.column() != 0:
            return super().data(index, role)

        item_data = item.data()
        if item_data is None:
            return super().data(index, role)

        item_type, id = item_data
        map = self._win.item_type_to_map(item_type)
        if len(map) == 0:
            return super().data(index, role)

        color = map[id].color
        return QColor(color[0], color[1], color[2])
    

class _Icon(QIcon):
    def __init__(self, filename):
        filename = os.path.join(os.path.dirname(__file__), "icons", filename)
        super().__init__(filename)


class _Separator(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)


class _MainWin(QMainWindow):
    def __init__(self):
        super().__init__()

        self.scale_factor = 1.0
        self.id_to_elem: dict[int, _Elem] = {}
        self.id_to_node: dict[int, _Node] = {}
        self.tree_view: QTreeView = None
        self.find_path_button: QPushButton = None
        self.graph = {}

        self._img_label: _ImgLabel = None
        self._img_qt: QImage = None
        self._scroll_area: QScrollArea = None
        self._status_bar: QStatusBar = None
        self._progress_bar: QProgressBar = None
        self._img_file_name: str = None
        self._item_model: _ItemModel = None
        self._nodes_item: QStandardItem = None
        self._detect_elements_button: QPushButton = None
        self._graph_buttons: list[QPushButton] = []
        self._create_graph_button: QPushButton = None
        self._model = None
        self._edges: dict[(int, int), _EdgeData] = {}

        self._create_win()
    

    def item_type_to_map(self, item_type: _ItemType):
        if item_type == _ItemType.NODE:
            return self.id_to_node
        else:
            return self.id_to_elem


    @property
    def edges(self):
        return self._edges


    @property
    def scroll_area(self):
        return self._scroll_area


    def _create_win(self):
        self.setWindowTitle("Floor Plan Recognition")
        self.setMinimumSize(500, 360)
        self.showMaximized()

        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(QAction(_Icon("new_file.svg"), "New", self, shortcut="Ctrl+N", 
            triggered=self._new_file))
        file_menu.addAction(QAction(_Icon("open_file.svg"), "Open...", self, shortcut="Ctrl+O", 
            triggered=self._open_file))
        file_menu.addAction(QAction(_Icon("save_graph.svg"), "Save graph...", self, shortcut="Ctrl+S", 
            triggered=self._save_graph))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Exit", self, shortcut="Ctrl+Q", triggered=self.close))

        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(QAction(_Icon("zoom_to_fit.svg"), "Zoom to fit", self, 
            shortcut="Ctrl+0", triggered=self._zoom_to_fit))
        view_menu.addAction(QAction(_Icon("show_100.svg"), "Show 100%", self, 
            shortcut="Ctrl+1", triggered=self._show_100))
        view_menu.addAction(QAction(_Icon("zoom_in.svg"), "Zoom in", self, 
            shortcut="Ctrl++", triggered=self._zoom_in))
        view_menu.addAction(QAction(_Icon("zoom_out.svg"), "Zoom out", self, 
            shortcut="Ctrl+-", triggered=self._zoom_out))

        v_layout = QVBoxLayout()

        widget = QWidget()
        widget.setLayout(v_layout)
        self.setCentralWidget(widget)

        h_layout = QHBoxLayout()
        h_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        v_layout.addLayout(h_layout)

        button = QPushButton()
        button.setIcon(_Icon("new_file.svg"))
        button.clicked.connect(self._new_file)
        button.setToolTip("New")
        h_layout.addWidget(button)

        button = QPushButton()
        button.setIcon(_Icon("open_file.svg"))
        button.clicked.connect(self._open_file)
        button.setToolTip("Open")
        h_layout.addWidget(button)

        button = QPushButton()
        button.setIcon(_Icon("save_graph.svg"))
        button.clicked.connect(self._save_graph)
        button.setToolTip("Save graph")
        h_layout.addWidget(button)

        h_layout.addWidget(_Separator())
        
        button = QPushButton()
        button.setIcon(_Icon("zoom_to_fit.svg"))
        button.clicked.connect(self._zoom_to_fit)
        button.setToolTip("Zoom to fit")
        h_layout.addWidget(button)

        button = QPushButton()
        button.setIcon(_Icon("show_100.svg"))
        button.clicked.connect(self._show_100)
        button.setToolTip("Show 100%")
        h_layout.addWidget(button)

        button = QPushButton()
        button.setIcon(_Icon("zoom_in.svg"))
        button.clicked.connect(self._zoom_in)
        button.setToolTip("Zoom in")
        h_layout.addWidget(button)
        h_layout.addWidget(button)

        button = QPushButton()
        button.setIcon(_Icon("zoom_out.svg"))
        button.clicked.connect(self._zoom_out)
        button.setToolTip("Zoom out")
        h_layout.addWidget(button)

        h_layout.addWidget(_Separator())

        button = QPushButton()
        button.setIcon(_Icon("add_node.svg"))
        button.clicked.connect(self._add_node)
        button.setToolTip("Add node")
        button.setEnabled(False)
        h_layout.addWidget(button)
        self._graph_buttons.append(button)

        button = QPushButton()
        button.setIcon(_Icon("remove_node.svg"))
        button.clicked.connect(self._remove_node)
        button.setToolTip("Remove node")
        button.setEnabled(False)
        h_layout.addWidget(button)
        self._graph_buttons.append(button)

        button = QPushButton()
        button.setIcon(_Icon("add_edge.svg"))
        button.clicked.connect(self._add_edge)
        button.setToolTip("Add edge")
        button.setEnabled(False)
        h_layout.addWidget(button)
        self._graph_buttons.append(button)

        button = QPushButton()
        button.setIcon(_Icon("remove_edge.svg"))
        button.clicked.connect(self._remove_edge)
        button.setToolTip("Remove edge")
        button.setEnabled(False)
        h_layout.addWidget(button)
        self._graph_buttons.append(button)

        h_layout.addWidget(_Separator())

        button = QPushButton()
        button.setIcon(_Icon("mark_as_exit.svg"))
        button.clicked.connect(self._mark_as_exit)
        button.setToolTip("Mark as exit")
        button.setEnabled(False)
        h_layout.addWidget(button)
        self._graph_buttons.append(button)

        button = QPushButton()
        button.setIcon(_Icon("clear_mark.svg"))
        button.clicked.connect(self._clear_mark)
        button.setToolTip("Clear mark")
        button.setEnabled(False)
        h_layout.addWidget(button)
        self._graph_buttons.append(button)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        v_layout.addWidget(splitter)

        self.tree_view = QTreeView()
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree_view.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self._context_menu)
        splitter.addWidget(self.tree_view)

        self._item_model = _ItemModel(self)
        self.tree_view.setModel(self._item_model)
        self._clear_list()

        selection_model = self.tree_view.selectionModel()
        selection_model.selectionChanged.connect(self._selection_changed)

        frame = QFrame()
        splitter.addWidget(frame)

        v_layout_2 = QVBoxLayout(frame)
        v_layout_2.setContentsMargins(0, 0, 0, 0)

        self._scroll_area = _ScrollArea(self)
        self._scroll_area.setBackgroundRole(QPalette.ColorRole.Dark)
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll_area.setVisible(True)

        self._img_label = _ImgLabel(self)
        self._img_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._img_label.setScaledContents(True)
        self._img_label.resize(300, 200)
        self._scroll_area.setWidget(self._img_label)
        v_layout_2.addWidget(self._scroll_area)

        self._status_bar = QStatusBar()
        v_layout_2.addWidget(self._status_bar)
        self._status_bar.setSizeGripEnabled(False)
        self._status_bar.hide()

        self._progress_bar = QProgressBar()
        self._status_bar.addPermanentWidget(self._progress_bar)

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

        self.find_path_button = QPushButton("Find path")
        self.find_path_button.clicked.connect(self.find_path)
        self.find_path_button.setEnabled(False)
        h_layout.addWidget(self.find_path_button)


    def _load_img(self, img_qt):
        self._img_qt = img_qt
        self._img_label.setPixmap(QPixmap.fromImage(img_qt))

    
    def _adjust_scroll_bar(self, scroll_bar: QScrollBar, pos, prev_pos):
        min_val = scroll_bar.minimum()
        max_val = scroll_bar.maximum()
        range = max_val - min_val
        range /= self.scale_factor
        val = prev_pos + pos*range
        scroll_bar.setValue(int(val))


    def scale_img(self, factor, point: QPointF=None):
        if self._img_qt is None:
            return
        
        if point is None:
            scroll_area_size = self._scroll_area.size()
            point = QPointF(scroll_area_size.width()/2, scroll_area_size.height()/2)

        old_scale = self.scale_factor
        new_scale = old_scale*factor

        scroll_bar_pos = QPointF(self._scroll_area.horizontalScrollBar().value(),
            self._scroll_area.verticalScrollBar().value())
        img_label_pos = QPointF(self._img_label.pos().x(), self._img_label.pos().y())
        delta_to_pos = (point - img_label_pos)/old_scale
        delta = delta_to_pos*(new_scale - old_scale)

        self._load_img(self._img_qt)
        self.scale_factor = new_scale
        self._img_label.resize(self.scale_factor * self._img_label.pixmap().size())

        self._scroll_area.horizontalScrollBar().setValue(int(scroll_bar_pos.x() + delta.x()))
        self._scroll_area.verticalScrollBar().setValue(int(scroll_bar_pos.y() + delta.y()))


    def _zoom_in(self):
        self.scale_img(1.25)


    def _zoom_out(self):
        self.scale_img(0.8)


    def _show_100(self):
        self.scale_factor = 1.0
        self._img_label.adjustSize()

    
    def _zoom_to_fit(self):
        src_size = self._img_label.size()
        trg_size = self._scroll_area.size()

        src_w = src_size.width()
        src_h = src_size.height()
        trg_w = trg_size.width() - self._scroll_area.verticalScrollBar().size().width()
        trg_h = trg_size.height() - self._scroll_area.horizontalScrollBar().size().height()

        if trg_w/trg_h < src_w/src_h:
            self.scale_img(trg_w/src_w)
        else:
            self.scale_img(trg_h/src_h)


    def _new_file(self):
        self._clear_list()
        self._img_label.clear()
        self._img_qt = None
        self._img_file_name = None
        self._detect_elements_button.setEnabled(False)
        self._create_graph_button.setEnabled(False)
        for button in self._graph_buttons:
            button.setEnabled(False)
        self.find_path_button.setEnabled(False)
        self.id_to_elem.clear()
        self.id_to_node.clear()
        self._edges.clear()
        self.graph.clear()


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
        self.scale_factor = 1.0
        self._img_label.adjustSize()
        self._zoom_to_fit()
        self._detect_elements_button.setEnabled(True)
        self._clear_list()

 
    def recalc_draw(self):
        for node in self.id_to_node.values():
            node.highlight_for_path = False
        for data in self._edges.values():
            data.highlight_for_path = False

        path_start_id = None
        for id, data in self.id_to_node.items():
            if data.is_selected:
                if path_start_id is None:
                    path_start_id = id
                else:
                    path_start_id = None
                    break
        
        for data in self._edges.values():
            if data.is_selected:
                path_start_id = None

        if path_start_id is not None:
            prev_path_id = None
            for path_id in self.id_to_node[path_start_id].path:
                self.id_to_node[path_id].highlight_for_path = True
                if prev_path_id is not None:
                    min_node_id = min(prev_path_id, path_id)
                    max_node_id = max(prev_path_id, path_id)
                    data = self._edges[min_node_id, max_node_id]
                    data.highlight_for_path = True
                prev_path_id = path_id


    def redraw(self):
        self.recalc_draw()
        self._img_label.update()
    

    def _save_graph(self):
        pass # FIXME
    

    def _clear_list(self):
        self._item_model.clear()
        self._item_model.setHorizontalHeaderLabels(("Elements", "Mark"))
        self._nodes_item = None
        self.tree_view.repaint()

    
    def _get_nodes_item(self):
        for item in self._item_model.findItems("Nodes", Qt.MatchFlag.MatchExactly):
            return item
        return None

    
    def _get_selected_childless_items(self, col, filter_item_type=None):
        selected_indexes = self.tree_view.selectedIndexes()
        indexes = set()

        for index in selected_indexes:
            item = self._item_model.itemFromIndex(index)

            if item.column() != col:
                continue

            if filter_item_type is not None:
                item_data = item.data()
                if item_data is None:
                    continue
                item_type, _ = item_data
                if item_type != filter_item_type:
                    continue

            child_items_num = item.rowCount()
            
            if child_items_num > 0:
                for i in range(child_items_num):
                    indexes.add(item.child(i).index())
            else:
                indexes.add(index)
        
        items = [self._item_model.itemFromIndex(index) for index in indexes]
        
        return items


    def _selection_changed(self):
        background = Image.open(self._img_file_name)

        items = self._get_selected_childless_items(0)

        for elem in self.id_to_elem.values():
            elem.is_selected = False

        for node in self.id_to_node.values():
            node.is_selected = False
            
        if len(items) == 0:
            self.recalc_draw()
            self._load_img(ImageQt.ImageQt(background))
            return

        panoptic_pred = self._output["panoptic_pred"].numpy()[0][:, :, np.newaxis]
        foreground_array = np.zeros((panoptic_pred.shape[0], panoptic_pred.shape[1], 4)).astype(np.uint8)

        for item in items:
            item_data = item.data()
            if item_data is None:
                continue
            item_type, id = item_data

            if item_type == _ItemType.NODE:
                self.id_to_node[id].is_selected = True
            else:
                self.id_to_elem[id].is_selected = True

                color = self.id_to_elem[id].color
                foreground_array += np.where(panoptic_pred == id,
                    (color[0], color[1], color[2], 200), (0, 0, 0, 0)).astype(np.uint8)

        foreground = Image.fromarray(foreground_array)
        background.paste(foreground, (0, 0), foreground)
        self.recalc_draw()
        self._load_img(ImageQt.ImageQt(background))

    
    def _get_common_edges(self, node_ids):
        edges = []

        for edge in self._edges.keys():
            if edge[0] in node_ids and edge[1] in node_ids:
                edges.append(edge)
        
        return edges
    

    def _get_possible_new_edges(self, node_ids):
        possible_new_edges = []

        node_ids = sorted(node_ids)
        common_edges = set(self._get_common_edges(node_ids))

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                edge = (node_ids[i], node_ids[j])
                if edge not in common_edges:
                    possible_new_edges.append(edge)

        return possible_new_edges


    def _context_menu(self, position):
        if not self.find_path_button.isEnabled():
            return

        node_items = self._get_selected_childless_items(0, _ItemType.NODE) 
        nodes_num = len(node_items)

        node_ids = [node_item.data()[1] for node_item in node_items]
        possible_new_edges_num = len(self._get_possible_new_edges(node_ids))

        menu = QMenu()
        menu.addAction(QAction("Add node", self, triggered=self._add_node))

        if nodes_num > 0:
            nodes_str = "nodes" if nodes_num > 1 else "node"
            menu.addAction(QAction(f"Remove {nodes_str}", self, triggered=self._remove_node))
            menu.addSeparator()
            if possible_new_edges_num > 0:
                edges_str = "edges" if possible_new_edges_num > 1 else "edge"
                menu.addAction(QAction(f"Add {edges_str}", self, triggered=self._add_edge))
                menu.addSeparator()
            menu.addAction(QAction("Mark as exit", self, triggered=self._mark_as_exit))
            menu.addAction(QAction("Clear mark", self, triggered=self._clear_mark))

        menu.exec(self.tree_view.viewport().mapToGlobal(position))


    def _get_exclude_colors(self):
        colors = [
            (0, 0, 0),
            (1, 1, 1),
            (_SELECTED_COLOR[0]/255, _SELECTED_COLOR[1]/255, _SELECTED_COLOR[2]/255),
            (_PATH_COLOR[0]/255, _PATH_COLOR[1]/255, _PATH_COLOR[2]/255),
        ]

        for node in self.id_to_node.values():
            color = (node.color[0]/255, node.color[1]/255, node.color[2]/255)
            colors.append(color)

        return colors


    def add_node_at_pos(self, pos):
        x = pos.x() / self.scale_factor
        y = pos.y() / self.scale_factor

        node_id = max(self.id_to_node.keys()) + 1 if len(self.id_to_node) > 0 else 1

        item1 = QStandardItem("Anonymous Node")
        item1.setEditable(True)
        item1.setData((_ItemType.NODE, node_id))

        item2 = QStandardItem("")
        item2.setEditable(False)
        item2.setData((_ItemType.NODE, node_id))

        parent = self._get_nodes_item()
        parent.appendRow((item1, item2))

        colors = distinctipy.get_colors(1, exclude_colors=self._get_exclude_colors(), rng=0)
        colors = (np.array(colors)*255).astype(np.uint8)
        self.id_to_node[node_id] = _Node(colors[0], item1, (x, y))

        self.clear_paths()
        self.redraw()

    
    def _add_node(self):
        center = self._scroll_area.viewport().rect().center()
        center = self._img_label.mapFromParent(center)
        self.add_node_at_pos(center)
    

    def _remove_node(self):
        items = self._get_selected_childless_items(0, _ItemType.NODE)
        if len(items) == 0:
            return

        nodes_str = "nodes" if len(items) > 1 else "node" 
        ret = QMessageBox.question(self, "Question", f"Are you sure you want to remove the selected {nodes_str}?", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret == QMessageBox.StandardButton.No:
            return

        for item in items:
            _, id = item.data()
            item.parent().removeRow(item.row())
            self.id_to_node.pop(id)

            edges_to_remove = []
            for edge in self.graph.keys():
                if edge[0] == id or edge[1] == id:
                    edges_to_remove.append(edge)
            
            for edge in edges_to_remove:
                self.graph.pop(edge)
                if edge in self._edges:
                    self._edges.pop(edge)
        
        self.clear_paths()
        self.redraw()

    
    def _add_edge(self):
        items = self._get_selected_childless_items(1, _ItemType.NODE)
        node_ids = [item.data()[1] for item in items]
        edges = self._get_possible_new_edges(node_ids)

        if len(edges) == 0:
            return

        for edge in edges:
            self._edges[edge] = _EdgeData()
            dist = self._calc_edge_dist(edge)
            self.graph[(edge[0], edge[1])] = dist
            self.graph[(edge[1], edge[0])] = dist
        
        self.clear_paths()
        self.redraw()


    def _remove_edge(self):
        edges = [edge for edge, data in self._edges.items() if data.is_selected]
        if len(edges) == 0:
            return

        edges_str = "edges" if len(edges) > 1 else "edge"
        ret = QMessageBox.question(self, "Question", f"Are you sure you want to remove the selected {edges_str}?", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret == QMessageBox.StandardButton.No:
            return

        for edge in edges:
            self._edges.pop(edge)
            self.graph.pop(edge)
            self.graph.pop((edge[1], edge[0]))

        self.clear_paths()
        self.redraw()

    
    def keyPressEvent(self, event: QKeyEvent)-> None:
        if event.key() == Qt.Key.Key_Delete.value:
            self._remove_node()
            self._remove_edge()
            return
        return super().keyPressEvent(event)
    
    
    def clear_paths(self):
        for node in self.id_to_node.values():
            node.path.clear()
     
    
    def _mark_as_exit(self):
        items = self._get_selected_childless_items(1, _ItemType.NODE)
        if len(items) == 0:
            return

        clear_paths = False

        for item in items:
            item.setText("Exit")
            _, id = item.data()
            self.id_to_node[id].mark = _Mark.EXIT
            clear_paths = True
        
        if clear_paths:
            self.clear_paths()

        self.redraw()
    
    
    def _clear_mark(self):
        items = self._get_selected_childless_items(1, _ItemType.NODE)
        if len(items) == 0:
            return

        clear_paths = False

        for item in items:
            item.setText("")
            _, id = item.data()
            self.id_to_node[id].mark = _Mark.NONE
            clear_paths = True

        if clear_paths:
            self.clear_paths()

        self.redraw()

    
    def _detect_elements_core(self):
        self._clear_list()
        self._clear_graph()
        self.id_to_elem.clear()

        if self._model is None:
            self._model = tf.saved_model.load("cubicasa5k/model")

        img_array = np.array(Image.open(self._img_file_name).convert("RGB"))

        self._output = self._model(tf.cast(img_array, tf.uint8))
        # self._output is a dict with keys: 
        # center_heatmap, instance_center_pred, instance_pred, 
        # panoptic_pred, offset_map, semantic_pred, 
        # semantic_logits, instance_scores, semantic_probs

        panoptic_pred = self._output["panoptic_pred"].numpy()[0]

        ids = np.unique(panoptic_pred)
        ids.sort()

        labels = np.unique(ids // _LABEL_DIVISOR)
        labels.sort()

        id_to_item = {}

        for label in labels:
            if label == ccl.Label.BACKGROUND:
                continue

            item_type = _label_to_item_type(label)
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

                id = label*_LABEL_DIVISOR + instance

                if instance == 0:
                    parent.setData((item_type, id))
                    id_to_item[id] = parent
                    continue

                item1 = QStandardItem(f"{label_str} {i}")
                item1.setEditable(True)
                item1.setData((item_type, id))
                id_to_item[id] = item1

                item2 = QStandardItem("")
                item2.setEditable(False)
                item2.setData((item_type, id))

                parent.appendRow((item1, item2))
        
        self.tree_view.expandAll()

        colors = distinctipy.get_colors(len(ids), exclude_colors=self._get_exclude_colors(), rng=0)
        colors = (np.array(colors)*255).astype(np.uint8)

        for id, color in zip(ids, colors):
            self.id_to_elem[id] = _Elem(color, id_to_item.get(id, None))

        self._create_graph_button.setEnabled(True)
        for button in self._graph_buttons:
            button.setEnabled(False)
        self.find_path_button.setEnabled(False)
        self.redraw()


    def _detect_elements(self):
        with _wait_cursor():
            self._detect_elements_core()


    def _calc_edge_dist(self, edge):
        center1 = self.id_to_node[edge[0]].center
        center2 = self.id_to_node[edge[1]].center

        return _euclidean_dist(center1, center2)
    

    def _clear_graph(self):
        if self._nodes_item is not None:
            self._nodes_item.removeRow(0)
            self._item_model.removeRow(self._nodes_item.row())
        self._nodes_item = None
        self.tree_view.repaint()

        self.id_to_node.clear()
        self.graph.clear()
        self._edges.clear()
    
        
    def _show_progress_bar(self, msg):
        self._status_bar.show()
        self._status_bar.showMessage(msg)
        self._status_bar.repaint()


    def _hide_progress_bar(self):
        self._progress_bar.reset()
        self._status_bar.hide()
        self._status_bar.clearMessage()
        self._status_bar.repaint()
    

    def _set_progress_bar_value(self, value):
        value = int(value)
        if value > self._progress_bar.value():
            self._progress_bar.setValue(value)
            self._progress_bar.repaint()
    

    def _create_graph_core(self):
        self._show_progress_bar("Creating graph...")
        self._clear_graph()

        panoptic_pred = self._output["panoptic_pred"].numpy()[0]
        instance_center_pred = self._output["instance_center_pred"].numpy()[0]

        elem_id_to_center_pred_max = {}
        elem_id_to_center = {}
        elem_id_graph = {}

        step = 1
        total_steps = panoptic_pred.shape[0]*panoptic_pred.shape[1]

        for i in range(panoptic_pred.shape[0]):
            for j in range(panoptic_pred.shape[1]):
                self._set_progress_bar_value(100*step/total_steps); step += 1

                elem_id = panoptic_pred[i][j]
                label = elem_id // _LABEL_DIVISOR
                if label not in (ccl.Label.ROOM, ccl.Label.DOOR):
                    continue

                if instance_center_pred[i][j] > elem_id_to_center_pred_max.get(elem_id, -1.0):
                    elem_id_to_center_pred_max[elem_id] = instance_center_pred[i][j]
                    elem_id_to_center[elem_id] = (j, i)

                neibs = _get_pixel_neibs(panoptic_pred, i, j)

                for neib in neibs:
                    if neib == elem_id:
                        continue
                    neib_label = neib // _LABEL_DIVISOR
                    if neib_label not in (ccl.Label.ROOM, ccl.Label.DOOR):
                        continue
                    if (elem_id, neib) not in elem_id_graph:
                        elem_id_graph[(elem_id, neib)] = 1.0

        elem_id_to_node_id = {}
        node_id = 0

        parent = QStandardItem("Nodes")
        parent.setEditable(False)
        self._nodes_item = parent
        self._item_model.appendRow(parent)

        for elem_id, elem in self.id_to_elem.items():
            label = elem_id // _LABEL_DIVISOR
            if label not in (ccl.Label.ROOM, ccl.Label.DOOR):
                continue

            node_id += 1

            label_str = self.id_to_elem[elem_id].item.text()

            item1 = QStandardItem(label_str)
            item1.setEditable(True)
            item1.setData((_ItemType.NODE, node_id))

            item2 = QStandardItem("")
            item2.setEditable(False)
            item2.setData((_ItemType.NODE, node_id))

            parent.appendRow((item1, item2))

            elem_id_to_node_id[elem_id] = node_id
            self.id_to_node[node_id] = _Node(elem.color, item1)

        self.tree_view.expand(parent.index())
        
        graph = {}

        for (elem_id, neib), dist in elem_id_graph.items():
            graph[(elem_id_to_node_id[elem_id], elem_id_to_node_id[neib])] = dist

        for elem_id, center in elem_id_to_center.items():
            self.id_to_node[elem_id_to_node_id[elem_id]].center = center
        
        for edge, _ in graph.items():
            dist = self._calc_edge_dist(edge)
            self.graph[edge] = dist
            min_node_id = min(edge[0], edge[1])
            max_node_id = max(edge[0], edge[1])
            self._edges[(min_node_id, max_node_id)] = _EdgeData()

        self._hide_progress_bar()
        
        for button in self._graph_buttons:
            button.setEnabled(True)
        self.find_path_button.setEnabled(True)
        self.redraw()

    
    def _create_graph(self):
        with _wait_cursor():
            self._create_graph_core()


    def _dijkstra(self, exit_id, id_to_neibs):
        id_to_dist = {id: sys.float_info.max for id in self.id_to_node.keys()}

        q = queue.PriorityQueue()

        id_to_dist[exit_id] = 0.0
        q.put((0.0, exit_id))

        while not q.empty():
            _, id = q.get()
            dist = id_to_dist[id]

            for neib in id_to_neibs[id]:
                cost = self.graph[(id, neib)]
                new_dist = dist + cost

                if new_dist < id_to_dist[neib]:
                    id_to_dist[neib] = new_dist
                    q.put((new_dist, neib))
        
        return id_to_dist


    def _calc_path(self, id: int, id_to_dist_dicts, exit_ids, id_to_neibs):
        best_exit_index = None
        min_dist = sys.float_info.max
        node = self.id_to_node[id]

        node.path = []

        for i, id_to_dist in enumerate(id_to_dist_dicts):
            dist = id_to_dist[id]
            if dist < min_dist:
                min_dist = dist
                best_exit_index = i
        
        if best_exit_index is None:
            return
        
        best_id_to_dist = id_to_dist_dicts[best_exit_index]
        path = [id]
        cur_id = id
        exit_id = exit_ids[best_exit_index]

        while cur_id != exit_id:
            min_dist = sys.float_info.max
            best_neib = None

            for neib in id_to_neibs[cur_id]:
                neib_dist = best_id_to_dist[neib]
                if neib_dist < min_dist:
                    min_dist = neib_dist
                    best_neib = neib
            
            if best_neib is None:
                return
            
            path.append(best_neib)
            cur_id = best_neib

        node.path = path


    def _find_paths_core(self, exit_ids):
        id_to_dist_dicts = []
        id_to_neibs = defaultdict(list)

        for (node_id, neib), _ in self.graph.items():
            id_to_neibs[node_id].append(neib)
        
        for exit_id in exit_ids:
            id_to_dist = self._dijkstra(exit_id, id_to_neibs)
            id_to_dist_dicts.append(id_to_dist)
        
        for id in self.id_to_node.keys():
            self._calc_path(id, id_to_dist_dicts, exit_ids, id_to_neibs)
    

    def find_path(self):
        exit_ids = []

        for id, data in self.id_to_node.items():
            if data.mark == _Mark.EXIT:
                exit_ids.append(id)

        if len(exit_ids) == 0:
            QMessageBox.critical(self, "Error", "No exit was set")
            return
        
        with _wait_cursor():
            self._find_paths_core(exit_ids)
        
        self.redraw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    win = _MainWin()
    win.show()

    sys.exit(app.exec())