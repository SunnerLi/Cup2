from config import *
from coder import *
import numpy as np
from config import *
import random
import os

def readScoreNetData():
    # Load name of files
    img_name_list = sorted(os.listdir(scorenet_img_path))
    dat_name_list = sorted(os.listdir(scorenet_dat_path))

    # Shuffle
    for i in range(len(img_name_list)/2):
        swap_index_1 = random.randint(0, len(img_name_list))
        swap_index_2 = random.randint(0, len(img_name_list))
        _ = img_name_list[swap_index_1]
        img_name_list[swap_index_1] = img_name_list[swap_index_2]
        img_name_list[swap_index_2] = _
        _ = dat_name_list[swap_index_1]
        dat_name_list[swap_index_1] = dat_name_list[swap_index_2]
        dat_name_list[swap_index_2] = _

    if len(img_name_list) != len(dat_name_list):
        print "file distribution is wrong..."

    # Create object
    img = cv2.imread(scorenet_img_path + img_name_list[0])    
    height, width, channel = np.shape(img)[0], np.shape(img)[1], np.shape(img)[2]
    x_train = np.ndarray([len(img_name_list), height, width, channel])
    y_train = np.ndarray([len(img_name_list),
        len(obj_name_2_index) * grid_height_num * grid_width_num  
    ])

    # Fill the list
    for i in range(len(img_name_list)):
        img = cv2.imread(scorenet_img_path + img_name_list[i])
        vector = encodeByFile(img, scorenet_dat_path + dat_name_list[i])
        x_train[i, ...] = img / 255.0
        y_train[i] = vector

    return (x_train, y_train)

def readUNetData(save_path='../img/unet/'):
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