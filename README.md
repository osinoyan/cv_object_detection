# CV Object Detection

Code for digit detection adapting YOLOv4.
 
 
## Environment
- Ubuntu 16.04 LTS

## Outline
1. [Installation](#Installation)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Training](#Training)


## Installation

### clone necessary repos
First, clone the [darknet repo](https://github.com/AlexeyAB/darknet)
Then, create another directory outside the *darknet/* and clone our [cv_object_detection repo](https://github.com/osinoyan/cv_object_detection)
Copy contents from *cv_object_detection/* to *darknet/* as the same file structure
```
$ git clone https://github.com/AlexeyAB/darknet
$ cd ..
$ git clone https://github.com/osinoyan/cv_object_detection
$ cd ..
$ cp -r cv_object_detection/* darknet/
```

### environment installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n cv_obj_det python=3.6
source activate cv_obj_det
pip install -r requirements.txt
```

## Dataset Preparation

### Prepare Images
Dataset is available from the share link of [google drive](https://drive.google.com/drive/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl)

Decompress these two file *test.zip*, *train.tar.gz* and then make sure to place the training and testing image files as the structure below (you must create the directories mannually):
```
    data
    +- obj
    |   +- 1.png (from train.tar.gz)
    |   +- ...
    +- test
    |   +- 1.png (from test.zip)
    |   +- ...
```

## Training
The settings for custom digit objects are prepared in our repo, you can train the object detector following the instructions below if you have placed the files correctly following [Installation](#Installation).
For more information, see [darknet repo](https://github.com/AlexeyAB/darknet).
### makefile
Build darknet.
```
cd darknet
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
make
```
### train models
Before training, download the pre-trained weights-file provided by darknet (162 MB): [yolov4.conv.137](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp)
To train models, run following commands.
```
./darknet detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137
```
### test
After training, try using the detector to test the prepared testing  data. The result will be writen to *result.json*
```
./darknet detector test data/obj.data cfg/yolov4-obj.cfg backup/yolov4-obj_final.weights -ext_output -dont_show -out result.json < data/test.txt
```