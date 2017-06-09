from keras.layers import UpSampling2D, concatenate, Cropping2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from config import *
import os

K.set_image_dim_ordering("tf")

class UNet():
    save_path = None
    def __init__(self, height, width, save_path='unet.h5'):
        input = Input((height, width, 3))

        # 1st down
        conv1 = Conv2D(8, (3, 3), activation='relu')(input)
        conv1 = Conv2D(8, (3, 3), activation='relu')(conv1)
        pool1 = MaxPooling2D()(conv1)

        # 2nd down
        conv2 = Conv2D(16, (3, 3), activation='relu')(pool1)
        conv2 = Conv2D(16, (3, 3), activation='relu')(conv2)
        pool2 = MaxPooling2D()(conv2)

        # 3rd down
        conv3 = Conv2D(32, (3, 3), activation='relu')(pool2)
        conv3 = Conv2D(32, (3, 3), activation='relu')(conv3)
        
        # 1st up
        up1 = UpSampling2D()(conv3)
        h, w = self.get_crop_shape(conv2, up1)
        crop1 = Cropping2D(cropping=((h, w)))(conv2) 
        dconv1 = concatenate([up1, crop1], axis=3)
        conv4 = Conv2D(16, (3, 3), activation='relu')(dconv1)
        conv4 = Conv2D(16, (3, 3), activation='relu')(conv4)

        # 2nd up
        up2 = UpSampling2D()(conv4)
        h, w = self.get_crop_shape(conv1, up2)
        crop2 = Cropping2D(cropping=((h, w)))(conv1)
        dconv1 = concatenate([up2, crop2], axis=3)
        conv5 = Conv2D(8, (3, 3), activation='relu')(dconv1)
        conv5 = Conv2D(8, (3, 3), activation='relu')(conv5)

        # Final crop
        h, w = self.get_crop_shape(input, conv5)
        print h, w
        padding = ZeroPadding2D(padding=(h[0], w[0]))(conv5)
        conv6 = Conv2D(1, (1, 1), activation='relu')(padding)

        # Load pre-trained model if exists
        self.model = Model(inputs=input, outputs=conv6)
        self.save_path = save_path 
        if os.path.exists(save_path):
            self.model.load_weights(save_path)
            print "<< UNet >> load unet pre-trained model..."
        print "<< UNet >> done initialize !"

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def compile(self):
        self.model.compile(
            loss='mse',
            optimizer='adam',
        )

    def train(self, x, y):
        self.model.fit(x, y, batch_size=4, epochs=general_epoch, verbose=1)
        self.model.save(self.save_path)

    def test(self, x):
        return self.model.predict(x, batch_size=1, verbose=0)

if __name__ == '__main__':
    model = UNet(270, 480)