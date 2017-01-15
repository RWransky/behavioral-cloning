from keras.models import Sequential
from keras import backend as K
import tensorflow as tf
from keras.constraints import *
from keras.layers.normalization import BatchNormalization
from keras.initializations import normal
from keras.models import Sequential, Model
from keras.engine.training import collect_trainable_weights
from keras.layers import *


# Originally needed image intensity normalization, but final model uses binary images so it is unneccessary.
def intensity_norm(data):
    return (data/float(255.0))


def custom_init(shape, name=None):
    return normal(shape, scale=0.01, name=name)


def build_network(input_shape=[40, 80, 1]):
    # input_shape is [n_rows, n_cols, num_channels] of the input images
    activation = 'relu'

    nrows, ncols, nchannels = input_shape

    past_image = Sequential()
    # past_image.add(Activation(activation=intensity_norm, input_shape=(nrows, ncols, nchannels)))

    # First three convolutional layers with kernal size 5x5 and stride 2x2
    past_image.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='valid', init='normal', activation=activation, input_shape=(nrows, ncols, nchannels)))
    past_image.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same'))
    past_image.add(Dropout(0.3))
    # past_image.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', init='normal', activation=activation))
    past_image.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', init='normal', activation=activation))
    past_image.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))

    # Final two convolutional layers with 3x3 kernal and no stride
    past_image.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), init='normal', activation=activation))
    past_image.add(Dropout(0.3))
    # past_image.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), init='normal', activation=activation))
    past_image.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same'))

    # Flatten convolition layers
    past_image.add(Flatten())
    past_image.add(Dense(10, init='normal', activation=activation))
    past_image.add(Dropout(0.3))
    past_image.add(Dense(5, init='normal', activation=activation))
    past_image.add(Dense(1))

    present_image = Sequential()
    # present_image.add(Activation(activation=intensity_norm, input_shape=(nrows, ncols, nchannels)))

    # First three convolutional layers with kernal size 5x5 and stride 2x2
    present_image.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='valid', init='normal', activation=activation, input_shape=(nrows, ncols, nchannels)))
    present_image.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same'))
    present_image.add(Dropout(0.3))
    # present_image.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', init='normal', activation=activation))
    present_image.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', init='normal', activation=activation))
    present_image.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))

    # Final two convolutional layers with 3x3 kernal and no stride
    present_image.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), init='normal', activation=activation))
    present_image.add(Dropout(0.3))
    # present_image.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), init='normal', activation=activation))
    present_image.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same'))

    # Flatten convolition layers
    present_image.add(Flatten())
    present_image.add(Dense(10, init='normal', activation=activation))
    present_image.add(Dropout(0.3))
    present_image.add(Dense(5, init='normal', activation=activation))
    present_image.add(Dense(1))

    # Model branch using past angle as input

    past_angle = Sequential()

    past_angle.add(Dense(512, init='normal', activation='tanh', input_dim=1))
    past_angle.add(Dropout(0.3))
    past_angle.add(Dense(32, init='normal', activation='tanh'))
    past_angle.add(Dense(1))

    # Combine three model branches into one

    model = Sequential()
    model.add(Merge([past_image, present_image, past_angle], mode='concat'))
    model.add(Dense(1))

    return model
