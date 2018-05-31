import pickle as pk
import numpy as np
import os
import keras
from keras.layers import Dense, LSTM, GRU, Bidirectional, Input, Conv2D, MaxPooling2D, Flatten, TimeDistributed, Reshape
from keras.layers import ELU, BatchNormalization
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils import to_categorical
from keras.models import Sequential

from sklearn.utils import shuffle

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

import pywt
import band_filter

SampFreq = 256
ChannelNum = 22

def load_data(data, label):
    t_data = np.load(open(data, 'rb')) 
    t_label = np.load(open(label, 'rb'))
    return t_data, t_label


def make_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, (1, 10), 
        data_format='channels_first', input_shape=(3,22,256)))
    model.add(BatchNormalization(axis=1, momentum=0.01))
    model.add(ELU())

#    model.add(Conv2D(64, (5, 10), data_format='channels_first'))
#    model.add(BatchNormalization(axis=1, momentum=0.01))
#    model.add(ELU())
#    model.add(MaxPooling2D((1, 2), data_format='channels_first'))
#
    model.add(Conv2D(64, (5, 10), data_format='channels_first'))
    model.add(BatchNormalization(axis=1, momentum=0.01))
    model.add(ELU())
    model.add(MaxPooling2D((1, 2), data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model

def dataset_preprocess(data, label):
   # data shuffle
   s_data, s_label = shuffle(data, label, random_state=2018)
   return s_data, s_label


def main():
    train_data, train_label = load_data(
            './LOPO/chb01/chb01-data-train.npy', 
            './LOPO/chb01/chb01-label-train.npy')
    val_data, val_label = load_data(
            './LOPO/chb01/chb01-data-val.npy', 
            './LOPO/chb01/chb01-label-val.npy')

    print('train_size:', np.shape(train_data))
    print('val_size:', np.shape(val_data))
    data_shape = np.shape(train_data[0])

    train_label = to_categorical(train_label)
    val_label = to_categorical(val_label)
    train_label = np.asarray(train_label)
    val_label = np.asarray(val_label)

    model = make_cnn_model(data_shape, 2)
    print model.summary()

    checkpoint = ModelCheckpoint(
            './model-LOPO.{epoch:04d}-{val_loss:.2f}.hdf5')

    logger = CSVLogger(os.path.join(".", "LOPO-training.log"))

    model.compile(
            loss='categorical_crossentropy', 
            optimizer = optimizers.SGD(
                lr=0.01, momentum=0.9, clipnorm=1., clipvalue=0.5),
            metrics=['accuracy'])
    model.fit(
            train_data, train_label, batch_size=32, 
            epochs=5, verbose=1, 
            validation_data=(val_data, val_label), shuffle=True, 
            initial_epoch=0, callbacks=[checkpoint, logger])


if __name__ == '__main__':
    main()
