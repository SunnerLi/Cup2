import _init_paths
from data_helper import readUNetData
from unet import UNet
from config import *
import numpy as np
import time
import cv2

# Read data
(train_x, train_y), (test_x, test_y) = readUNetData() 
model = UNet(img_height, img_width, save_path=model_path+unet_model_name)

# Test
_time = time.time()
for i in range(1):
    prediction = model.test(train_x)
print "process time: ", time.time() - _time

# Show
cv2.imshow('ground truth', train_y[0])
cv2.imshow('prediction', prediction[0])
cv2.imshow('ground truth1', train_y[1])
cv2.imshow('prediction1', prediction[1])
cv2.waitKey(0)