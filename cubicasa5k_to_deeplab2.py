import os
import numpy as np
from tqdm import tqdm
from xml.dom import minidom
from skimage.draw import polygon
from PIL import Image
import cv2


semantic_labels = {
    "Wall" : 1,
    "Railing" : 1,
    "Door" : 2,
    "Window" : 3,
}


def get_points(e):
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
            
            if attr in semantic_labels:
                X, Y = get_points(e)
                rr, cc = polygon(X, Y)
                cc, rr = self._clip_outside(cc, rr)
                annotation[cc, rr, 0] = semantic_labels[attr]

        img = Image.fromarray(annotation)
        img.save(self._get_annotation_file_path())


def create_annotation_files(set_name, files_num):
    dataset_dir_path = os.path.join("datasets", "cubicasa5k")
    txt_file_path = os.path.join(dataset_dir_path, set_name + ".txt")
    
    print(f"Creating annotation files for {set_name} set")
    
    with open(txt_file_path) as f:
        for sample_dir_name in tqdm(f.readlines()[:files_num]):
            sample_dir_name = os.path.normpath(sample_dir_name[1:-1])
            sample_dir_path = os.path.join(dataset_dir_path, sample_dir_name)
            annotation_creator = AnnotationCreator(sample_dir_path)
            annotation_creator.create()

def main():
    create_annotation_files("train", 420)
    create_annotation_files("val", 40)
    create_annotation_files("test", 40)
    

if __name__ == '__main__':
    main()