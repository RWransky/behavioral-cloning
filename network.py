from keras.models import Sequential
from highway_unit import *
from keras.constraints import *
from keras.layers.normalization import BatchNormalization


def build_network(n_layers=10, dim=32, input_shape=[20, 80, 3], n_classes=1 shared=0):
    # input_shape is [n_rows, n_cols, num_channels] of the input images
    activation = 'relu'

    trans_bias = -n_layers // 10

    shared_highway = HighwayUnit(dim, 3, 3, activation=activation, transform_bias=trans_bias, border_mode='same')

    def _highway():
        if shared == 1:
            return shared_highway
        return HighwayUnit(dim, 3, 3, activation=activation, transform_bias=trans_bias, border_mode='same')

    nrows, ncols, nchannels = input_shape
    model = Sequential([
        Reshape((1, nrows, ncols, nchannels), input_shape=(nrows*ncols*nchannels,)),
        Dropout(0.2),
        Convolution2D(dim, 5, 5, activation=act)
    ])

    model.add(BatchNormalization())

    for i in range(2):
        model.add(_highway())
        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(n_classes, init='he_normal'))

    return model
