from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras import backend as K
from config import *
import os

K.set_image_dim_ordering('tf')

class ScoreNet(object):
    save_path = None
    def __init__(self, save_path='47.h5', use_VGG=False):
        self.model = Sequential()

        
        if not use_VGG:
            # 1st Conv
            self.model.add(Conv2D(5, (16, 16), activation='relu', input_shape=(270, 480, 3)))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))

            # 2nd Conv
            self.model.add(Conv2D(10, (8, 8), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))

            # 3rd Conv
            self.model.add(Conv2D(15, (4, 4), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))

            # 4th Conv
            self.model.add(Conv2D(20, (2, 2), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            # FC
            self.model.add(Flatten())
            self.model.add(Dense(scorenet_fc_num, activation='relu'))
        
        else:
            self.model.add(ZeroPadding2D((1,1),input_shape=(270, 480, 3)))
            self.model.add(Convolution2D(8, 3, 3, activation='relu', name='conv1_1'))
            self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Convolution2D(8, 3, 3, activation='relu', name='conv1_2'))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Convolution2D(16, 3, 3, activation='relu', name='conv2_1'))
            self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Convolution2D(16, 3, 3, activation='relu', name='conv2_2'))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Convolution2D(32, 3, 3, activation='relu', name='conv3_1'))
            self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Convolution2D(32, 3, 3, activation='relu', name='conv3_2'))
            self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Convolution2D(32, 3, 3, activation='relu', name='conv3_3'))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Convolution2D(64, 3, 3, activation='relu', name='conv4_1'))
            self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Convolution2D(64, 3, 3, activation='relu', name='conv4_2'))
            self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Convolution2D(64, 3, 3, activation='relu', name='conv4_3'))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Convolution2D(64, 3, 3, activation='relu', name='conv5_1'))
            self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Convolution2D(64, 3, 3, activation='relu', name='conv5_2'))
            self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Convolution2D(64, 3, 3, activation='relu', name='conv5_3'))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            self.model.add(Flatten(name="flatten"))
            self.model.add(Dense(512, activation='relu', name='dense_1'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(scorenet_fc_num, name='dense_3'))
            self.model.add(Activation("relu",name="final_layer"))

        # Load model if exist
        self.save_path = save_path
        if os.path.exists(save_path):
            print "<< ScoreNet >> load score net pre-trained model..."
            self.model = load_model(save_path)
        print "<< ScoreNet >> done initialize..."
        
    def compile(self):
        self.model.compile(
            loss='mse',
            optimizer='adam',
        )

    def train(self, x, y):
        self.model.fit(x, y, batch_size=32, epochs=general_epoch, verbose=1)
        self.model.save(self.save_path)

    def test(self, x):
        return self.model.predict(x, batch_size=1, verbose=0)
