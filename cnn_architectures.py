from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
import matplotlib.pyplot as plt

def nvidia_model(row,col,ch):
    
    # CNN architecture by NVIDIA, ref: https://arxiv.org/pdf/1604.07316v1.pdf
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch))) # Normalization
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

def custom_model(row,col,ch):
    model = Sequential()
    
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch))) # Normalization
    model.add(Convolution2D(24,7,7,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,1,1,activation="relu"))
    model.add(Flatten())
    model.add(Dense(300))
    model.add(Dense(200))
    model.add(Dense(100))
    model.add(Dense(1))

    return model