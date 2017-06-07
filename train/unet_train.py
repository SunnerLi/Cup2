import _init_paths
from data_helper import readUNetData
from unet import UNet
from config import *
import numpy as np
import cv2

# Read data
(train_x, train_y), (test_x, test_y) = readUNetData() 
model = UNet(img_height, img_width, save_path=model_path + unet_model_name)
train_x.astype('float32')
train_y.astype('float32')

# Train
model.compile()
model.train(train_x, train_y)