from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2019)

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
import argparse

SampFreq = 256
ChannelNum = 22

def get_parser():
    parser = argparse.ArgumentParser(description='train multiple LOPO jobs')
    parser.add_argument('-f', '--folder', help='data floder')
    parser.add_argument('-s', '--save_dir', help='save dir')
    parser.add_argument('-se', '--start_epoch', help='start epoch')
    parser.add_argument('-e', '--epoch', help='train epoch')
    parser.add_argument('-c', '--ckpt_file', help='ckpt file')
    return parser.parse_args()

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
    model.add(Dropout(0.2, seed=2018))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def dataset_preprocess(data, label):
   # data shuffle
   s_data, s_label = shuffle(data, label, random_state=2018)
   return s_data, s_label


def main():
    args = get_parser()
    data_path = args.folder
    save_dir = args.save_dir
    start_epoch = int(args.start_epoch)
    epoch = int(args.epoch)
    ckpt_path = args.ckpt_file

    train_data, train_label = load_data(
            os.path.join(
                data_path, '{}-data-train.npy'.format(data_path[-5:])), 
            os.path.join(
                data_path, '{}-label-train.npy'.format(data_path[-5:])))

    val_data, val_label = load_data(
            os.path.join(
                data_path, '{}-data-val.npy'.format(data_path[-5:])), 
            os.path.join(
                data_path, '{}-label-val.npy'.format(data_path[-5:])))

    print('train_size:', np.shape(train_data))
    print('val_size:', np.shape(val_data))
    data_shape = np.shape(train_data[0])

    train_label = to_categorical(train_label)
    val_label = to_categorical(val_label)
    train_label = np.asarray(train_label)
    val_label = np.asarray(val_label)
    if not ckpt_path:
        model = make_cnn_model(data_shape, 2)
        model.compile(
            loss='categorical_crossentropy', 
                optimizer = optimizers.SGD(
                lr=0.01, momentum=0.9, clipnorm=1., clipvalue=0.5),
                metrics=['accuracy'])
    else:
        model = load_model(ckpt_path)

    print(model.summary())
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    checkpoint = ModelCheckpoint(
        os.path.join(save_dir, 'model.{epoch:04d}-{val_loss:.2f}.hdf5'))

    logger = CSVLogger(os.path.join(save_dir, "training-{}-{}.log.csv".format(start_epoch, epoch)))

    model.fit(
        train_data, train_label, batch_size=32, 
        epochs=epoch, verbose=1, 
        validation_data=(val_data, val_label), shuffle=True, 
        initial_epoch=start_epoch, callbacks=[checkpoint, logger])

if __name__ == '__main__':
    main()
