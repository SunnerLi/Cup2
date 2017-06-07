import _init_paths
from data_helper import readScoreNetData
from scorenet import ScoreNet

# Read data
(x_train, y_train) = readScoreNetData() 
model = ScoreNet(save_name='../model/scorenet.h5')
x_train.astype('float32')
y_train.astype('float32')

# Train
model.compile()
model.train(x_train, y_train)