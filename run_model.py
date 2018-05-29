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

def load_data():
    d_data = np.load(open('image_cnn_data.npy', 'rb'))
    d_label = np.load(open('cnn_label.npy', 'rb'))
    return d_data, d_label



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



def make_model(time_step, input_shape, num_classes):
#    in_list = []
#    out_list = []
#    digit_input = Input(shape=input_shape)
#    x = Conv2D(64, (1, 3), data_format='channels_first')(digit_input)
#    x = Conv2D(64, (22, 3), data_format='channels_first')(x)
#    x = MaxPooling2D((1, 2), data_format='channels_first')(x)
#    out = Flatten()(x)
#
#    vision_model = Model(digit_input, out)
#    vision_model.summary()
#    # Then define the tell-digits-apart model
#    for i in range(time_step):
#        image_a = Input(shape = input_shape)
#        out_a = vision_model(image_a)
#        out_list.append(out_a)
#        in_list.append(image_a)
#
#    concatenated = keras.layers.concatenate(out_list)
#    out = Dense(1, activation='sigmoid')(concatenated)
#
#    classification_model = Model(in_list, out)
#    return classification_model
    model = Sequential()
    model.add(TimeDistributed(Conv2D(64, (1, 10), data_format='channels_first'),
        input_shape=(6, 3, 22, 256)))
    model.add(TimeDistributed(Conv2D(64, (5, 10), data_format='channels_first')))
    model.add(TimeDistributed(MaxPooling2D((1, 2), data_format='channels_first')))


    model.add(Reshape((6, -1)))
    model.add(GRU(
        units=10, kernel_initializer='orthogonal', 
        bias_initializer='ones', dropout=0.5, recurrent_dropout=0.5))

    model.add(Dense(num_classes, activation='softmax'))
    return model
    
#    model = Sequential()
#

##    model.add(Dense(256, activation='softmax'))
##    model.add(GRU(units=100, dropout=0.2))
##    model.add(Bidirectional(LSTM(10, return_sequences = True), input_shape = (49,1)))

#    model.add(Bidirectional(GRU(10, return_sequences = False)))
#    model.add(Dense(64))
#    model.add(Dense(num_classes, activation='softmax'))
#
#    model.summary()
#    return model
def dataset_preprocess(data, label):
   # data shuffle
   s_data, s_label = shuffle(data, label, random_state=2018)
   return s_data, s_label


def main():
    x, y = load_data()
    x, y = dataset_preprocess(x, y)

    print(np.shape(x))
    print(np.shape(y))
    data_shape = np.shape(x[0])

    y = to_categorical(y)
    print y[:10]
    y = np.asarray(y)
    x = np.asarray(x)
#    model = make_model(6, (3, 22, 256), 2)
    model = make_cnn_model(data_shape, 2)
    print model.summary()

    checkpoint = ModelCheckpoint(
            './model.{epoch:04d}-{val_loss:.2f}.hdf5')

    logger = CSVLogger(os.path.join(".", "training.log"))

    model.compile(
            loss='categorical_crossentropy', 
            optimizer = optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1., clipvalue=0.5),
            metrics=['accuracy'])
    model.fit(
            x, y, batch_size=32, 
            epochs=5, verbose=1, 
            validation_split=0.1, shuffle=True, 
            initial_epoch=0, callbacks=[checkpoint, logger])


if __name__ == '__main__':
    main()
