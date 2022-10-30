import os
import random
from absl import logging, flags, app

flags.DEFINE_string("R2V_root", 
    default=None, 
    help="R2V dataset root folder.",
    required=True)

FLAGS = flags.FLAGS


_DATASET_SPLIT_SIZES = {
    "train" : 1023,
    "val" : 128,
    "test" : 128,
}

    
def _get_img_paths(R2V_root):
    img_paths = []
    for root, dirs, files in os.walk(R2V_root):
        rel_root = os.path.relpath(root, R2V_root)
        for file in files:
            if file.endswith(".jpg"):
                img_paths.append(os.path.join(rel_root, file))
    return img_paths


def _split_dataset(R2V_root):
    img_paths = _get_img_paths(R2V_root)

    random.seed(0)
    random.shuffle(img_paths)

    start_i = 0
    for dataset_split in ("train", "val", "test"):
        samples_num = _DATASET_SPLIT_SIZES[dataset_split]
        samples = img_paths[start_i:start_i+samples_num]
        start_i = samples_num

        txt_file = os.path.join(R2V_root, dataset_split + ".txt")
        with open(txt_file, "w+") as f:
            for sample in samples:
                f.write(sample + "\n")


def main(_):
    logging.get_absl_handler().setFormatter(None)
    _split_dataset(FLAGS.R2V_root)


if __name__ == '__main__':
    app.run(main)  