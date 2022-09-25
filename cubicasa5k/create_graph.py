from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from google.protobuf import text_format
from deeplab2 import config_pb2

flags.DEFINE_string(
    "config_file",
    default=None,
    help="Proto file which specifies the experiment configuration.",
    required=True)

FLAGS = flags.FLAGS


def main(_):
    logging.info('Reading the config file.')
    with tf.io.gfile.GFile(FLAGS.config_file, 'r') as proto_file:
        config = text_format.ParseLines(proto_file, config_pb2.ExperimentOptions())


if __name__ == '__main__':
    app.run(main)