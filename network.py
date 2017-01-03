from keras.models import Sequential
from keras import backend as K
from highway_unit import *
from keras.constraints import *
from keras.layers.normalization import BatchNormalization


def intensity_norm(data):
    return (data/float(255.0) - 0.5)


def build_network(input_shape=[80, 260, 2]):
    # input_shape is [n_rows, n_cols, num_channels] of the input images
    activation = 'relu'

    nrows, ncols, nchannels = input_shape
    model = Sequential()
    model.add(Activation(activation=intensity_norm, input_shape=(nrows, ncols, nchannels)))

    # First three convolutional layers with kernal size 5x5 and stride 2x2
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', init='normal', activation=activation))
    model.add(Dropout(0.3))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', init='normal', activation=activation))
    model.add(Dropout(0.3))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', init='normal', activation=activation))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same'))

    # Final two convolutional layers with 3x3 kernal and no stride
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), init='normal', activation=activation))
    model.add(Dropout(0.3))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), init='normal', activation=activation))
    model.add(Dropout(0.3))

    # Flatten convolition layers
    model.add(Flatten())
    model.add(Dense(25, init='he_normal', activation=activation))
    model.add(Dense(10, init='he_normal', activation=activation))
    model.add(Dense(2))

    # model.add(Dense(1000, activation=activation))
    # model.add(Dense(500, activation=activation))
    # model.add(Dense(n_classes, init='he_normal', activation='softmax'))

    # Three fully connected layers
    # model.add(Dense(100, activation=activation))
    # model.add(Dropout(0.5))
    # model.add(Dense(50, activation=activation))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation=activation))
    # model.add(Dropout(0.5))
    # model.add(Dense(n_classes))
    # model.add(Convolution2D(dim, 3, 3, activation=activation))
    # model.add(Dropout(0.2))

    # model.add(BatchNormalization())

    # model.add(Convolution2D(64, 3, 3, activation=activation))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))

    # model.add(Convolution2D(256, 3, 3, activation=activation))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))

    # model.add(Convolution2D(256, 3, 3, activation=activation))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    # for i in range(2):
    #     model.add(_highway())
    #     model.add(BatchNormalization())
    #     model.add(Dropout(0.5))

    # for i in range(2):
    #     model.add(_highway())
    #     model.add(Dropout(0.5))
    #     model.add(BatchNormalization())

    # model.add(Dropout(0.3))

    # model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))

    # model.add(Flatten())
    # model.add(Dense(256, activation=activation))
    # model.add(Dense(n_classes))

    return model
