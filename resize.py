import _init_paths
from lib.config import *
import cv2
import os

img_path_dir = './xml/'

def resize(img_path_dir):
    for name in os.listdir(img_path_dir):
        if name[-4:] == '.bmp':
            img = cv2.imread(img_path_dir + name)
            img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            cv2.imwrite(img_path_dir + name, img)

def addPrefix(img_path_dir, prefix_sentence):
    for name in os.listdir(img_path_dir):
        if name[-4:] == '.xml':
            os.rename(img_path_dir + name, img_path_dir + prefix_sentence + name)

def deletePrefix(img_path_dir):
    for name in os.listdir(img_path_dir):
        if name[-4:] == '.xml':
            os.rename(img_path_dir + name, img_path_dir + name[7:])

addPrefix(img_path_dir, 'video1_')