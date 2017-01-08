from keras.models import Sequential
from keras import backend as K
from highway_unit import *
from keras.constraints import *
from keras.layers.normalization import BatchNormalization


def intensity_norm(data):
    return (data/float(255.0) - 0.5)


def build_network(n_layers=6, dim=32, input_shape=[80, 260, 2]):
    # input_shape is [n_rows, n_cols, num_channels] of the input images
    activation = 'relu'

    trans_bias = -n_layers // 10

    nrows, ncols, nchannels = input_shape
    model = Sequential()
    model.add(Activation(activation=intensity_norm, input_shape=(nrows, ncols, nchannels)))
    model.add(Convolution2D(32, 5, 5, activation=activation))
    model.add(Dropout(0.3))
    model.add(HighwayUnit(dim, 3, 3, activation=activation, transform_bias=trans_bias, border_mode='same'))

    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', init='normal', activation=activation))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same'))

    # Final two convolutional layers with 3x3 kernal and no stride
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), init='normal', activation=activation))
    model.add(Dropout(0.3))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), init='normal', activation=activation))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))

    # Flatten convolition layers
    model.add(Flatten())
    model.add(Dense(2))

    return model
