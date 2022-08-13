import os
import numpy as np
from tqdm import tqdm
from xml.dom import minidom
from skimage.draw import polygon
from PIL import Image
import cv2

_SEMANTIC_LABELS = {
    "Wall" : 1,
    "Railing" : 1,
    "Door" : 2,
    "Window" : 3,
}

_USE_DATASET_RATIO = 0.05


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


class AnnotationCreator:   
    def __init__(self, sample_dir_path):
        self.sample_dir_path = sample_dir_path
        self.width = None
        self.height = None
    
    
    def _get_img_file_path(self):
        return os.path.join(self.sample_dir_path, "F1_scaled.png")
     
    
    def _get_svg_file_path(self):
        return os.path.join(self.sample_dir_path, "model.svg")
    
    
    def _get_annotation_file_path(self):
        return os.path.join(self.sample_dir_path, "annotation.png")
    
    
    def _clip_outside(self, rr, cc):
        s = np.column_stack((rr, cc))
        s = s[s[:, 0] < self.height]
        s = s[s[:, 1] < self.width]
    
        return s[:, 0], s[:, 1]


    def create(self):
        fplan = cv2.imread(self._get_img_file_path())
        fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
        self.height, self.width, nchannel = fplan.shape
        
        annotation = np.zeros((self.height, self.width, nchannel), dtype=np.uint8)
        
        svg = minidom.parse(self._get_svg_file_path())
        
        for e in svg.getElementsByTagName('g'):
            attr = e.getAttribute("id")
            
            if attr in _SEMANTIC_LABELS:
                X, Y = _get_points(e)
                rr, cc = polygon(X, Y)
                cc, rr = self._clip_outside(cc, rr)
                annotation[cc, rr, 0] = _SEMANTIC_LABELS[attr]

        img = Image.fromarray(annotation)
        img.save(self._get_annotation_file_path())


def _prepare_dataset_split(dataset_split):
    print("Processing dataset split {}".format(dataset_split))
    
    dataset_dir_path = os.path.join("datasets", "cubicasa5k")
    txt_file_path = os.path.join(dataset_dir_path, dataset_split + ".txt")
    
    with open(txt_file_path) as f:
        lines = f.readlines()
        samples_num = _USE_DATASET_RATIO*len(lines)
        for sample_dir_name in tqdm(lines[:samples_num]):
            sample_dir_name = os.path.normpath(sample_dir_name[1:-1])
            sample_dir_path = os.path.join(dataset_dir_path, sample_dir_name)
            
            annotation_creator = AnnotationCreator(sample_dir_path)
            annotation_creator.create()

def main():
    for dataset_split in ("train", "val"):
        _prepare_dataset_split(dataset_split)
    

if __name__ == '__main__':
    main()
