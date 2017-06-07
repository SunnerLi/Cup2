import _init_paths
from scorenet import ScoreNet
from config import *
import numpy as np
import coder
import time
import cv2

img_dir = '../img/scorenet/frame090.bmp'

# Test
model = ScoreNet(save_path='../model/scorenet.h5')
model.compile()
x_test = np.expand_dims(cv2.imread(img_dir), axis=0)
_time = time.time()
prediction = model.test(x_test)
print 'time comsumption: ', time.time() - _time

# Show the test result
res = coder.decodeByVector(x_test[0], prediction)
cv2.imshow('show', res)
cv2.waitKey(0)