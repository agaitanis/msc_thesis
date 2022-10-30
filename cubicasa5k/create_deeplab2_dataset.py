from collections import defaultdict
import os
import numpy as np
import cv2
import tensorflow as tf
import shutil
import cubicasa5k.labels as ccl
from tqdm import tqdm
from xml.dom import minidom
from skimage.draw import polygon
from PIL import Image
from absl import app
from absl import flags
from absl import logging
from enum import Enum
from collections import defaultdict

flags.DEFINE_string("cubicasa5k_root", 
    default=None, 
    help="CubiCasa5k dataset root folder.",
    required=True)

flags.DEFINE_string("output_dir",
    default=None,
    help="Path to save dataset for deeplab2.",
    required=True)

FLAGS = flags.FLAGS


_DATASET_SPLIT_SIZES = {
    "train" : 4200,
    "val" : 400,
    "test" : 400,
}

_MAX_IMG_SIZE = 1024


class Channel(int, Enum):
    R = 0
    G = 1
    B = 2


def _create_img_array(img_file_path):
    img_array = cv2.imread(img_file_path)
    
    return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # correct color channels


def _get_points(e):
    pol = next(p for p in e.childNodes if p.nodeName == "polygon")
    points = pol.getAttribute("points").split(' ')
    points = points[:-1]

    X, Y = np.array([]), np.array([])
    for a in points:
        x, y = a.split(',')
        X = np.append(X, np.round(float(x)))
        Y = np.append(Y, np.round(float(y)))

    return X, Y


    
def _clip_outside(rr, cc, height, width):
    s = np.column_stack((rr, cc))
    s = s[s[:, 0] < height]
    s = s[s[:, 1] < width]

    return s[:, 0], s[:, 1]


def _create_labels_array(img_array, svg_file_path):
    instance_ids = defaultdict(int)
    height, width, nchannel = img_array.shape
    labels_array = np.zeros((height, width, nchannel), dtype=np.uint8)
    
    svg = minidom.parse(svg_file_path)
    
    for e in svg.getElementsByTagName('g'):
        svg_id = e.getAttribute("id")

        if svg_id in ("Wall", "Railing", "Window", "Door"):
            label = ccl.svg_label_to_label[svg_id]
            if svg_id == "Door":
                instance_ids[label] += 1
        elif "Space " in e.getAttribute("class"):
            label = e.getAttribute("class").split(" ")[1]
            label = ccl.svg_label_to_label[label]
            instance_ids[label] += 1
        else:
            continue

        X, Y = _get_points(e)
        rr, cc = polygon(X, Y)
        cc, rr = _clip_outside(cc, rr, height, width)

        instance_id = instance_ids[label]
        labels_array[cc, rr, Channel.R] = label
        labels_array[cc, rr, Channel.G] = instance_id // 256
        labels_array[cc, rr, Channel.B] = instance_id % 256
    
    return labels_array


def _save_as_png(array, file_path):
    img = Image.fromarray(array)
    img.save(file_path)


def _resize_image(array, method):
    height, width, _ = array.shape
    if height <= _MAX_IMG_SIZE and width <= _MAX_IMG_SIZE:
        return array

    if method == tf.image.ResizeMethod.BILINEAR:
        array = array.astype(np.float64)
    array = np.expand_dims(array, axis=0)

    size = (_MAX_IMG_SIZE, _MAX_IMG_SIZE)
    array = tf.image.resize(array, size=size, method=method, preserve_aspect_ratio=True)

    array = np.squeeze(array, axis=0)
    if method == tf.image.ResizeMethod.BILINEAR:
        array = array.astype(np.uint8)
    
    return array


def _create_sample(sample_dir_path, new_sample_dir_path):
    img_file_path = os.path.join(sample_dir_path, "F1_scaled.png")
    svg_file_path = os.path.join(sample_dir_path, "model.svg")
    
    new_img_file_path = os.path.join(new_sample_dir_path, "image.png")
    labels_file_path = os.path.join(new_sample_dir_path, "labels.png")
    
    img_array = _create_img_array(img_file_path)
    labels_array = _create_labels_array(img_array, svg_file_path)

    img_array = _resize_image(img_array, tf.image.ResizeMethod.BILINEAR)
    labels_array = _resize_image(labels_array, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    _save_as_png(img_array, new_img_file_path)
    _save_as_png(labels_array, labels_file_path)
    

def _create_dataset(cubicasa5k_root, output_dir, dataset_split):
    logging.info("Creating dataset split {}".format(dataset_split))
    
    txt_file_path = os.path.join(cubicasa5k_root, dataset_split + ".txt")
    shutil.copy(txt_file_path, output_dir)
    
    with open(txt_file_path) as f:
        samples_num = _DATASET_SPLIT_SIZES[dataset_split]
        
        for sample_dir_name in tqdm(f.readlines()[:samples_num]):
            sample_dir_name = os.path.normpath(sample_dir_name[1:-1])
            sample_dir_path = os.path.join(cubicasa5k_root, sample_dir_name)
            new_sample_dir_path = os.path.join(output_dir, sample_dir_name)
            tf.io.gfile.makedirs(new_sample_dir_path)
            
            _create_sample(sample_dir_path, new_sample_dir_path)


def main(_):
    logging.get_absl_handler().setFormatter(None)

    try:
        shutil.rmtree(FLAGS.output_dir)
    except:
        pass
    tf.io.gfile.makedirs(FLAGS.output_dir)
    
    for dataset_split in ("train", "val", "test"):
        _create_dataset(FLAGS.cubicasa5k_root, FLAGS.output_dir, dataset_split)
    

if __name__ == '__main__':
    app.run(main)
