import os
from tqdm import tqdm
import torchvision


def create_annotation_file(sample_dir_path : str) -> None:
    image_file_path = os.path.join(sample_dir_path, "F1_scaled.png")
    fplan = torchvision.io.read_image(image_file_path)
    channels, height, width = fplan.shape


def create_annotation_files(set_name : str) -> None:
    txt_file_path = os.path.join("datasets", "cubicasa5k", set_name + ".txt")
    
    print(f"Creating annotation files for {set_name} set")
    
    with open(txt_file_path) as f:
        for sample_dir_name in tqdm(f.readlines()):
        # for sample_dir_name in f.readlines():
            sample_dir_name = os.path.normpath(sample_dir_name[1:-1])
            sample_dir_path = os.path.join("datasets", "cubicasa5k", sample_dir_name)
            create_annotation_file(sample_dir_path)


def main():
    # create_annotation_files("train")
    # create_annotation_files("val")
    create_annotation_files("test")
    

if __name__ == '__main__':
    main()