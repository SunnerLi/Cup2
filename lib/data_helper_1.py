from config import *
import numpy as np
import cv2
import os

def readData(save_path='img/'):
    # Read Training data
    train_x, train_y = readDataSingleFolder(save_path + 'train/')

    # Read Testing data
    test_x, test_y = readDataSingleFolder(save_path + 'test/')

    return (train_x, train_y), (test_x, test_y)

def readDataSingleFolder(save_path):
    img_name_list = sorted(os.listdir(save_path))
    if len(img_name_list) % 2 == 1:
        print "image number error..."
        exit()
    x = np.ndarray([len(img_name_list) / 2, img_height, img_width, img_channel])
    y = np.ndarray([len(img_name_list) / 2, img_height, img_width, 1])
    for i in range(len(img_name_list)):
        if img_name_list[i][-5] == 'g':
            y[i/2] = np.expand_dims(cv2.imread(save_path + img_name_list[i], 0), -1)
        else:
            x[i/2] = cv2.imread(save_path + img_name_list[i], 1)
    return x, y

# (train_x, train_y), (test_x, test_y) = readData()