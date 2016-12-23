import time
import os

from keras.optimizers import *
from keras.callbacks import *

from network import *
from data_helpers import *


def train():
    print('Loading data...')
    train_dataset, valid_dataset, train_angles, valid_angles = get_training_data()
    train_angles = np.uint8(train_angles)
    valid_angles = np.uint8(valid_angles)
    print('Size of training dataset is: {} samples'.format(train_dataset.shape[0]))
    print('Size of validation dataset is: {} samples'.format(valid_dataset.shape[0]))

    # define loss function and training metric

    loss = 'mean_squared_error'
    metric = 'acc'

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
    fParams = '{0}/{1}.hdf5'.format(path, saving)
    saveParams = ModelCheckpoint(fParams, monitor='val_loss', save_best_only=True)

    callbacks = [saveParams]

    his = model.fit(train_dataset, train_angles,
                    validation_data=(valid_dataset, valid_angles),
                    nb_epoch=20, batch_size=100,
                    callbacks=callbacks)

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
