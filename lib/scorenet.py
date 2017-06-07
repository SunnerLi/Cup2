from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras import backend as K
from config import *
import os

K.set_image_dim_ordering('tf')

class ScoreNet(object):
    save_path = None
    def __init__(self, save_path='scorenet.h5'):
        self.model = Sequential()

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
        self.model.fit(x, y, batch_size=8, epochs=general_epoch, verbose=1)
        self.model.save(self.save_path)

    def test(self, x):
        return self.model.predict(x, batch_size=1, verbose=0)
