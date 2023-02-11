# Route planning for emergency evacuation using graph traversal algorithms

Automatic identification of various design elements in a floor plan image has gained increased attention in 
recent research. Current work aims to extract information from a floor plan image and transform it into a 
graph which is used for path finding in an emergency evacuation. First, the basic elements of the floor plan 
image, i.e. walls, rooms and doors are identified. This is achieved using 
[Panoptic-Deeplab](https://github.com/google-research/deeplab2/blob/main/g3doc/projects/panoptic_deeplab.md) 
which is a state-of-the-art deep neural network for panoptic segmentation of images and it is available in 
[DeepLab2](https://github.com/google-research/deeplab2), 
an image segmentation library.  The neural network was trained using 
[CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k), 
a large-scale floor plan image dataset, containing 5000 samples, annotated into over 80 floor plan object 
categories. Then, using the prediction of each pixel, a graph is created which shows how the rooms and the 
doors are connected to each other. An application was developed which presents this information in a 
user-friendly manner and provides edit capabilities of the graph. Finally, the exits are set and the optimal 
path for evacuation is calculated from each node using the Dijkstra algorithm.

## Installation

The current project was implemented using cuda 11.7 and cudnn 8.5.0.\
Install cuda and cudnn using the instructions in:\
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

Clone the current repository:
```bash
git clone https://github.com/agaitanis/msc_thesis.git ${PROJECT_DIR}
```

Create a new conda environment:
```bash
conda create --name ${ENV_NAME} python=3.8.13
conda activate ${ENV_NAME}
```

Install the following libraries:
```bash
pip install tensorflow==2.7.0 keras==2.7.0 cython==0.29.32 protobuf==3.20.1 opencv-python==4.6.0.66 tqdm==4.64.1 scikit-image==0.19.3 numpy==1.23.1 PyQt6==6.4.0 distinctipy==1.2.2
```

Compile pycocotools:
```bash
cd ${PROJECT_DIR}/cocoapi/PythonAPI
make
```

Add libaries to PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:${PROJECT_DIR}
export PYTHONPATH=$PYTHONPATH:${PROJECT_DIR}/models
export PYTHONPATH=$PYTHONPATH:${PROJECT_DIR}/cocoapi/PythonAPI
```

Compile protobuf:
```bash
cd ${PROJECT_DIR}
protoc deeplab2/*.proto --python_out=.
```

Compile custom ops:
```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
OP_NAME='deeplab2/tensorflow_ops/kernels/merge_semantic_and_instance_maps_op'

# GPU support (https://www.tensorflow.org/guide/create_op#compiling_the_kernel_for_the_gpu_device)
nvcc -std=c++14 -c -o ${OP_NAME}_kernel.cu.o ${OP_NAME}_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

g++ -std=c++14 -shared -o ${OP_NAME}.so ${OP_NAME}.cc ${OP_NAME}_kernel.cc \
  ${OP_NAME}_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -lcuda ${TF_LFLAGS[@]}
```
If you get an error like this:
```bash
fatal error: third_party/gpus/cuda/include/cuda_fp16.h: No such file or directory
```
then do the following:
```bash
# Find the directory where tensorflow is installed
pip show tensorflow

# Go to the following directory or create one if it does not exit
cd ${TENSORFLOW_DIR}/include/third_party/gpus/

# Symlink your CUDA include directory here:
ln -s ${CUDA_DIR} ./cuda
```

To test if the compilation is done successfully, you can run:
```bash
python deeplab2/tensorflow_ops/python/kernel_tests/merge_semantic_and_instance_maps_op_test.py
```

To test if DeepLab2 is successfully installed and configured, you can run:
```bash
# Model training test (test for custom ops, protobuf)
python deeplab2/model/deeplab_test.py

# Model evaluator test (test for other packages such as orbit, cocoapi, etc)
python deeplab2/trainer/evaluator_test.py
```

## Dataset preparation

Download CubiCasa5K from [here](https://zenodo.org/record/2613548#.Y-e33NJBy0k)
and place it in the folder datasets/cubicasa5k.

Convert the dataset to the format that is required by DeepLab2:
```bash
python cubicasa5k/create_deeplab2_dataset.py --cubicasa5k_root=datasets/cubicasa5k/ --output_dir=datasets/deeplab2/cubicasa5k/
```

Create the TFRecords:
```bash
python deeplab2/data/build_cubicasa5k_data.py --cubicasa5k_root=datasets/deeplab2/cubicasa5k/ --output_dir=datasets/deeplab2/cubicasa5k/tf_records
```

## Model training

Download the pretrained checkpoints from [here](https://github.com/google-research/deeplab2/blob/main/g3doc/projects/imagenet_pretrained_checkpoints.md)
and place them in deeplab2/initial_checkpoints/.

Train your model using the following command (the best model configuration is used in this example):
```bash
python deeplab2/trainer/train.py --config_file=deeplab2/configs/cubicasa5k/panoptic_deeplab/59_wide_resnet41.textproto --mode=train_and_eval --model_dir=results --num_gpus=1 >& results/59.txt
```

Export the model in order to be used by the tool:
```bash
python deeplab2/export_model.py --experiment_option_path=deeplab2/configs/cubicasa5k/panoptic_deeplab/59_wide_resnet41.textproto --checkpoint_path=results/59/ckpt-40000 --output_path=tool/model
```

## Tool usage

Open the tool using the following command:
```bash
python tool/tool.py
```

Screenshot of the tool:
![alt text](https://github.com/agaitanis/msc_thesis/blob/main/tool/screenshot.png)

* Image manipulation
	* Open an image with File > Open.
	* Zoom by pressing the zoom buttons or by scrolling.
	* Move the picture by pressing Shift + Left Click.
* Delect elements
	* Press the "Detect elements" button to detect the floorplan elements using the exported model.
	* Select the items on the list to draw the predicted floorplan elements on the picture.
* Create graph
	* Press the "Create graph" button to automatically create the graph of the rooms/doors layout.
* Edit graph
	* Select a node/edge by clicking on it in the picture.
	* Select multiple nodes/edges by pressing Ctrl.
	* Move a node by pressing Shift + Left Click on the node.
	* Create a new node by pressing the "New node" button or by pressing Right Click > New node here.
	* Create a new edge by selecting two nodes and then pressing the "New edge" button.
	* Delete a node/edge by pressing the "Delete" button or by pressing the Del key.
* Calculate paths
	* Set one or more exits by pressing the "Mark as exit" button
	* Press the "Calculate paths" button in order to calculate the paths using the Dijkstra algorithm.
	* Select one node to show the shortest path to the nearest exit.
* Save graph
	* Save the graph to xml with File > Save graph.


Example:
![alt text](https://github.com/agaitanis/msc_thesis/blob/main/tool/example.png)

## References

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
