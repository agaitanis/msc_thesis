import os
import numpy as np
import cv2
import torch
import tensorflow as tf
import shutil
from tqdm import tqdm
from xml.dom import minidom
from skimage.draw import polygon
from PIL import Image
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_string("cubicasa5k_root", None, "CubiCasa5k dataset root folder.",
                    required=True)
flags.DEFINE_string("output_dir", None, "Path to save dataset for deeplab2.",
                    required=True)

_SEMANTIC_LABELS = {
    "Wall" : 1,
    "Railing" : 2,
    "Door" : 3,
    "Window" : 4,
}

_DATASET_SPLIT_SIZES = {
    "train" : 42, # FIXME Change to 420
    "val" : 4, # FIXME Change to 40
    "test" : 4, # FIXME Change to 40
}

_TARGET_SIZE = (256, 256)


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
    height, width, nchannel = img_array.shape
    labels_array = np.zeros((height, width, nchannel), dtype=np.uint8)
    
    svg = minidom.parse(svg_file_path)
    
    for e in svg.getElementsByTagName('g'):
        attr = e.getAttribute("id")
        
        if attr in _SEMANTIC_LABELS:
            X, Y = _get_points(e)
            rr, cc = polygon(X, Y)
            cc, rr = _clip_outside(cc, rr, height, width)
            labels_array[cc, rr, 0] = _SEMANTIC_LABELS[attr]
    
    return labels_array


def _save_as_png(array, file_path):
    img = Image.fromarray(array)
    img.save(file_path)


def _resize_array(array, size, mode):
    array = np.moveaxis(array, -1, 0)
    if mode == "bilinear":
        array = array.astype(np.float64)
    tensor = torch.from_numpy(array)
    tensor = tensor.unsqueeze(0)

    # TODO Try tf.image.resize, tf.image.resize_with_pad
    tensor = torch.nn.functional.interpolate(tensor, size=size, mode=mode)

    tensor = tensor.squeeze(0)
    array = tensor.numpy()
    if mode == "bilinear":
        array = array.astype(np.uint8)
    array = np.moveaxis(array, 0, -1)
    
    return array
        

def _create_sample(sample_dir_path, new_sample_dir_path):
    img_file_path = os.path.join(sample_dir_path, "F1_scaled.png")
    svg_file_path = os.path.join(sample_dir_path, "model.svg")
    
    new_img_file_path = os.path.join(new_sample_dir_path, "image.png")
    labels_file_path = os.path.join(new_sample_dir_path, "labels.png")
    
    img_array = _create_img_array(img_file_path)
    labels_array = _create_labels_array(img_array, svg_file_path)
    
    img_array = _resize_array(img_array, _TARGET_SIZE, "bilinear")
    labels_array = _resize_array(labels_array, _TARGET_SIZE, "nearest")
    
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


def main(unused_argv):
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
