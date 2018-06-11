import pickle as pk
import numpy as np
import os
import keras
from keras.layers import Dense, LSTM, GRU, Bidirectional, Input, Conv2D, MaxPooling2D, Flatten, TimeDistributed, Reshape
from keras.layers import ELU, BatchNormalization, Dropout
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils import to_categorical
from keras.models import Sequential, load_model

from sklearn.utils import shuffle

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV


def make_cnn_model(input_shape, num_classes):
    model = Sequential()
    percent_noise = 0.1
    noise = (1.0/255) * percent_noise
    model.add(GaussianNoise(noise, input_shape=(3,22,768)))
    model.add(Conv2D(64, (1, 30), 
        data_format='channels_first'))
    model.add(BatchNormalization(axis=1, momentum=0.01))
    model.add(ELU())
    model.add(Dropout(0.2, seed=2018))

#    model.add(Conv2D(64, (5, 10), data_format='channels_first'))
#    model.add(BatchNormalization(axis=1, momentum=0.01))
#    model.add(ELU())
#    model.add(MaxPooling2D((1, 2), data_format='channels_first'))
#
    model.add(Conv2D(64, (5, 30), data_format='channels_first'))
    model.add(BatchNormalization(axis=1, momentum=0.01))
    model.add(ELU())
    model.add(MaxPooling2D((1, 6), data_format='channels_first'))
    model.add(Dropout(0.2, seed=2018))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model

