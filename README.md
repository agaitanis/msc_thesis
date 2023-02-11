# msc_thesis: Route Planning for Emergency Evacuation using graph traversal algorithms

Abstract:
Automatic identification of various design elements in a floor plan image has gained 
increased attention in recent research. Current work aims to extract information from a floor 
plan image and transform it into a graph which is used for path finding in an emergency 
evacuation. First, the basic elements of the floor plan image, i.e. walls, rooms and doors 
are identified. This is achieved using Panoptic-Deeplab which is a state-of-the-art deep 
neural network for panoptic segmentation of images and it is available in DeepLab2, an 
image segmentation library. The neural network was trained using CubiCasa5K, a large-scale 
floor plan image dataset, containing 5000 samples, annotated into over 80 floor plan 
object categories. Then, using the prediction of each pixel, a graph is created which shows 
how the rooms and the doors are connected to each other. An application was developed 
which presents this information in a user-friendly manner and provides edit capabilities of 
the graph. Finally, the exits are set and the optimal path for evacuation is calculated from 
each node using the Dijkstra algorithm.

## Installation

### References

1. Bowen Cheng, Maxwell D. Collins, Yukun Zhu, Ting Liu, Thomas S. Huang, Hartwig
Adam, and Liang-Chieh Chen. Panoptic-deeplab: A simple, strong, and fast baseline for
bottom-up panoptic segmentation. CoRR, abs/1911.10194, 2019.

2. Mark Weber, Huiyu Wang, Siyuan Qiao, Jun Xie, Maxwell D. Collins, Yukun Zhu,
Liangzhe Yuan, Dahun Kim, Qihang Yu, Daniel Cremers, Laura Leal-Taixé, Alan L.
Yuille, Florian Schroff, Hartwig Adam, and Liang-Chieh Chen. Deeplab2: A tensorflow
library for deep labeling. CoRR, abs/2106.09748, 2021.

3. Ahti Kalervo, Juha Ylioinas, Markus Häikiö, Antti Karhu, and Juho Kannala. Cubicasa5k:
A dataset and an improved multi-task model for floorplan image analysis. CoRR,
abs/1904.01920, 2019.
