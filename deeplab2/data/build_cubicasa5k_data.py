import math
import os
import shutil

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import numpy as np

from deeplab2.data import data_utils
from deeplab2.data import dataset

from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('cubicasa5k_root', None, 'CubiCasa5k dataset root folder.',
                    required=True)

flags.DEFINE_string('output_dir', None,
                    'Path to save converted TFRecord of TensorFlow examples.',
                    required=True)

_SPLITS_TO_SIZES = dataset.CUBICASA5K_INFORMATION.splits_to_sizes
_LABEL_DIVISOR = dataset.CUBICASA5K_INFORMATION.panoptic_label_divisor


# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}
_PANOPTIC_LABEL_FORMAT = 'raw'

_DATASET_SPLIT_MAP = {
    "train" : 4200,
    "val" : 400,
    "test" : 400,
}

_NUM_SHARDS_MAP = {
    "train" : 100,
    "val" : 10,
    "test" : 10,
}


def _get_images(cubicasa5k_root, dataset_split):
    """Gets files for the specified data type and dataset split.
    
    Args:
      cubicasa5k_root: String, path to CubiCasa5k dataset root folder.
      dataset_split: String, dataset split ('train', 'val', 'test')
    
    Returns:
      A list of sorted file names or None when getting label for
        test set.
    """
    txt_file_path = os.path.join(cubicasa5k_root, dataset_split + ".txt")
    
    filenames = []
    
    samples_num = _DATASET_SPLIT_MAP[dataset_split]
    
    with open(txt_file_path) as f:
        for sample_dir_name in f.readlines()[:samples_num]:
            sample_dir_name = os.path.normpath(sample_dir_name[1:-1])
            sample_dir_path = os.path.join(cubicasa5k_root, sample_dir_name)
            filenames.append(os.path.join(sample_dir_path, "image.png"))
    
    return filenames


def _get_image_name(image_path):
    path, _ = os.path.split(image_path)
    path, dir2 = os.path.split(path)
    _, dir1 = os.path.split(path)
    return dir1 + "_" + dir2


def _get_panoptic_annotation(image_path):
    dir_path = os.path.dirname(image_path)
    return os.path.join(dir_path, "labels.png")


def _generate_panoptic_label(panoptic_annotation_file):
    with tf.io.gfile.GFile(panoptic_annotation_file, 'rb') as f:
        panoptic_label = data_utils.read_image(f.read())
    
    panoptic_label = np.array(panoptic_label, dtype=np.int32)
    semantic_label = panoptic_label[:, :, 0]
    instance_label = (panoptic_label[:, :, 1] * 256 + panoptic_label[:, :, 2])

    panoptic_label = semantic_label * _LABEL_DIVISOR + instance_label

    return panoptic_label.astype(np.int32)


def _create_panoptic_label(image_path):
    panoptic_annotation_file = _get_panoptic_annotation(image_path)
    panoptic_label = _generate_panoptic_label(panoptic_annotation_file)

    return panoptic_label.tobytes(), _PANOPTIC_LABEL_FORMAT


def _convert_dataset(cubicasa5k_root, dataset_split, output_dir):
    """Converts the specified dataset split to TFRecord format.
    
    Args:
      cubicasa5k_root: String, path to CubiCasa5k dataset root folder.
      dataset_split: String, the dataset split (one of `train`, `val` and `test`).
      output_dir: String, directory to write output TFRecords to.
    """
    image_files = _get_images(cubicasa5k_root, dataset_split)

    num_images = len(image_files)
    expected_dataset_size = _SPLITS_TO_SIZES[dataset_split]
    if num_images != expected_dataset_size:
        raise ValueError('Expects %d images, gets %d' %
                         (expected_dataset_size, num_images))

    num_shards = _NUM_SHARDS_MAP[dataset_split]
    num_per_shard = int(math.ceil(len(image_files) / num_shards))

    for shard_id in range(num_shards):
        logging.info('Creating shard %d of %d.', shard_id+1, num_shards)
        shard_filename = '%s-%05d-of-%05d.tfrecord' % (
            dataset_split, shard_id, num_shards)
        output_filename = os.path.join(output_dir, shard_filename)
        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in tqdm(range(start_idx, end_idx)):
                # Read the image.
                with tf.io.gfile.GFile(image_files[i], 'rb') as f:
                    image_data = f.read()
    
                if dataset_split == 'test':
                    label_data, label_format = None, None
                else:
                    label_data, label_format = _create_panoptic_label(image_files[i])
                 
                # Convert to tf example.
                image_name = _get_image_name(image_files[i])
                example = data_utils.create_tfexample(image_data,
                                                      _DATA_FORMAT_MAP['image'],
                                                      image_name, label_data,
                                                      label_format)
                 
                tfrecord_writer.write(example.SerializeToString())


def main(unused_argv):
    logging.get_absl_handler().setFormatter(None)

    try:
        shutil.rmtree(FLAGS.output_dir)
    except:
        pass
    tf.io.gfile.makedirs(FLAGS.output_dir)
    
    for dataset_split in ('train', 'val', 'test'):
        logging.info('Starts to processing dataset split %s.', dataset_split)
        _convert_dataset(FLAGS.cubicasa5k_root, dataset_split, FLAGS.output_dir)


if __name__ == '__main__':
    app.run(main)
