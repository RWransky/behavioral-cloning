import time
import os
import numpy as np

from keras.optimizers import *
from keras.callbacks import *

from network import *
from data_helpers import *
from model_helpers import *


def train():
    print('Loading data...')
    train_dataset, valid_dataset, train_angles, valid_angles = get_training_data()
    # Intensity normalize images
    train_dataset = intensity_normalization(train_dataset)
    valid_dataset = intensity_normalization(valid_dataset)
    # Collect normalizing values per channel
    # train_mean_red = np.mean(train_dataset[:, :, 0])
    # train_std_red = np.std(train_dataset[:, :, 0])

    # train_mean_green = np.mean(train_dataset[:, :, 1])
    # train_std_green = np.std(train_dataset[:, :, 1])

    # train_mean_blue = np.mean(train_dataset[:, :, 2])
    # train_std_blue = np.std(train_dataset[:, :, 2])

    # np.save('train_mean_red', train_mean_red)
    # np.save('train_std_red', train_std_red)
    # np.save('train_mean_green', train_mean_green)
    # np.save('train_std_green', train_std_green)
    # np.save('train_mean_blue', train_mean_blue)
    # np.save('train_std_blue', train_std_blue)
    # # Normalize imaging data
    # train_mean = [train_mean_red, train_mean_green, train_mean_blue]
    # train_std = [train_std_red, train_std_green, train_std_blue]
    # for i in range(3):
    #     train_dataset[:, :, i] = (train_dataset[:, :, i] - train_mean[i])/train_std[i]
    #     valid_dataset[:, :, i] = (valid_dataset[:, :, i] - train_mean[i])/train_std[i]

    # train_angles = np.uint8(train_angles)
    # valid_angles = np.uint8(valid_angles)
    print('Size of training dataset is: {} samples'.format(train_dataset.shape[0]))
    print('Size of validation dataset is: {} samples'.format(valid_dataset.shape[0]))

    # train_angles = np.expand_dims(train_angles, -1)
    # valid_angles = np.expand_dims(valid_angles, -1)

    # define loss function and training metric

    # loss = 'sparse_categorical_crossentropy'
    loss = 'mean_squared_error'
    metric = 'mean_squared_error'

    print('Building network ...')
    model = build_network()
    model.summary()

    model.compile(optimizer='adam',
                  loss=loss, metrics=[metric])

    print('Training...')
    path = './training'
    if not os.path.exists(path):
        os.makedirs(path)

    start_time = time.time()

    saving = 'highway_model'
    fParams = '{0}/{1}.h5'.format(path, saving)
    saveParams = ModelCheckpoint(fParams, monitor='val_mean_squared_error', save_best_only=True)

    callbacks = [saveParams]

    his = model.fit(train_dataset, train_angles,
                    validation_data=(valid_dataset, valid_angles),
                    nb_epoch=5, batch_size=16, callbacks=callbacks)

    # save model to json file
    if not os.path.exists('./models'):
        os.makedirs('./models')
    fModel = open('models/' + saving + '.json', 'w')
    json_str = model.to_json()
    fModel.write(json_str)

    end_time = time.time()

    print('training time: %.2f' % (end_time - start_time))


def main():
    train()

if __name__ == "__main__":
    main()
