# Cup2

<p align="center">
  <img src="https://raw.githubusercontent.com/SunnerLi/Cup2/master/img/readme_img/cup_recog.gif?token=AK99R1f9GVRjjumWtjfFQAIUJOVFj0Wvks5ZQpqAwA%3D%3D" width=480 height=270/>
</p> 


Abstract
---

This is the final program of computer vision. First, the program should draw the boundary of the gate, flower or other building. Second, the program should draw the bounding box toward each object. Next, the program might capture the butterfly. The last acquirement is recognizing the types of the object.    

Idea
---
![](https://github.com/SunnerLi/Cup2/blob/master/img/readme_img/cup_structure.jpg)

<br/>

The whole tasks can be divide as two parts: segmentation and classification. This program refer the UNet structure to build a convolutional neural network with downsample and upsample processes. This model can get the segmentation graph in advance. Next, the thinking of YOLO is used to build the convolutional neural network that the grids should predict the object scores. 

In the implementation of scoring model, there're two structure we have provided: simple one and simplified VGG16. You can set the `use_VGG` flag to switch the structure. 


Usage
---
Train the segmentation model:
```
$ cd train
$ python unet_train.py
```

Train the classification model:
```
$ cd train
$ python scorenet.py
```

Run demo:
```
$ python main.py
```

Train by Your Image
---
#### UNet
The training data of UNet model are laid under the `img/unet/train` folder. You should follow the name rule. 
```
The name of original image: <the_index_of_image>.png
The name of ground truth  : <the_index_of_image>.g.png
```
Especially, the `index_of_image` should be the same between the original image and its ground truth. 

<br/>

#### ScoreNet
The training image are located under the `img/scorenet/` folder, and the corresponding description file are located under the `xml/` folder. There is no any rule of naming. But you should double check the name of each pair should be the same. For example:
```
The example name of image       : SAMPLE_NAME.bmp
The corresponding name should be: SAMPLE_NAME.xml
```
After you ready to put your files, generate the `.dat` description file by the following command to ensure the number of `.dat` file should be the same as the images. 
```
$ cd .
$ rm * dat/
$ python parse.py
```

How to Label
---
There's no program can help you to label the training data of UNet. However, you can use [this](https://github.com/tzutalin/labelImg) to generate the `.xml` files toward ScoreNet. The label program can generate the PASCAL-format xml file which is the only acceptable format in this project. 