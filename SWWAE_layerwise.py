import keras
from unpooling_layer import Unpooling 

import pickle as pk
import numpy as np
import os
import keras
from keras.layers import Dense, LSTM, GRU, Bidirectional, Input, Conv2D, MaxPooling2D, Flatten, TimeDistributed, Reshape
from keras.layers import ELU, BatchNormalization, Dropout, AveragePooling2D, UpSampling2D, Conv2DTranspose, Concatenate, Reshape
from keras.layers.convolutional import SeparableConv2D
from keras.applications.mobilenet import DepthwiseConv2D
from keras.constraints import max_norm

from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.initializers import he_uniform, he_normal, glorot_normal
import keras.backend as K
from sklearn.utils import shuffle

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='train multiple win LOPO jobs')
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

def SWWAE_model(data_shape, num_classes):
    input_tensor = Input(shape=data_shape, name='Input')
    x = Conv2D(16, (1, 32), 
            data_format='channels_first', padding='same', use_bias=False,
            kernel_initializer= glorot_normal())(input_tensor)
    

    x = BatchNormalization(axis=1, momentum=0.01)(x)
    
    x = DepthwiseConv2D((22, 1), 
        data_format='channels_first', use_bias=False,
        depth_multiplier=2, depthwise_constraint=max_norm(1.),
        kernel_initializer=glorot_normal())(x)

    x = BatchNormalization(axis=1, momentum=0.01)(x)
    x = ELU()(x)

    orig_1 = x
    x = AveragePooling2D((1, 4), data_format='channels_first')(x)
    encode_1 = Dropout(0.5)(x)

    # decode part
    
    x = UpSampling2D(size=(1, 4))(encode_1)
    the_shape = K.int_shape(orig_1)
    shape = (1, the_shape[1], the_shape[2], the_shape[3])
    origReshaped = Reshape(shape)(orig_1)
    # print('origReshaped.shape: ' + str(K.int_shape(origReshaped)))
    xReshaped = Reshape(shape)(x)
    # print('xReshaped.shape: ' + str(K.int_shape(xReshaped)))
    together = Concatenate(axis=1)([origReshaped, xReshaped])
    # print('together.shape: ' + str(K.int_shape(together)))
    x = Unpooling()(together)
    x = ELU()(x)
    x = BatchNormalization(axis=1, momentum=0.01)(x)


    x = Conv2DTranspose(32, (22, 1), 
        data_format='channels_first', use_bias=False,
        kernel_initializer=glorot_normal())(x)

    x = BatchNormalization(axis=1, momentum=0.01)(x)
    out_1 = Conv2DTranspose(3, (1, 32), 
            data_format='channels_first', padding='same', use_bias=False,
            kernel_initializer= glorot_normal())(x)

    #---------------------------------------------------------
    x = SeparableConv2D(32, (1, 16), 
        kernel_initializer= glorot_normal(), use_bias=False,
        padding='same',
        data_format='channels_first')(encode_1)

    x = BatchNormalization(axis=1, momentum=0.01)(x)
    x = ELU()(x)
    orig_2 = x
    x = AveragePooling2D((1, 8), data_format='channels_first')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    out_2 = Dense(num_classes, activation='softmax', 
        kernel_constraint=max_norm(0.25))(x)
    model = Model(inputs=[input_tensor], outputs=[out_1, out_2])
    return model


def dataset_preprocess(data, label):
   # data shuffle
   s_data, s_label = shuffle(data, label, random_state=2018)
   return s_data, s_label


def main():
    data_path = './floyd-data/chb02'
    save_dir = '.'
    start_epoch = 0
    epoch = 5

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
    model = SWWAE_model(data_shape, 2)
    model.compile(loss=['','categorical_crossentropy',
        optimizer = optimizers.Adagrad(), 
        metrics = ['accuracy'])
    print model.summary()
    model.fit(
        train_data, [train_data, train_label], batch_size=32, 
        epochs=epoch, verbose=1, 
        validation_data=(val_data, [val_data, val_label]), shuffle=True, 
        initial_epoch=start_epoch)

if __name__ == '__main__':
    main()
