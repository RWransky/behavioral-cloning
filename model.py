import time
import os
import numpy as np

from keras.optimizers import *
from keras.callbacks import *

from network import *
from data_helpers import *


def train():
    # define loss function and training metric

    loss = 'mse'
    metric = 'mse'

    print('Building network ...')
    model = build_network()
    model.summary()

    model.compile(optimizer='adam',
                  loss=loss, metrics=[metric])

    # save model to json file
    saving = 'highway_model'
    if not os.path.exists('./models'):
        os.makedirs('./models')
    fModel = open('models/' + saving + '.json', 'w')
    json_str = model.to_json()
    fModel.write(json_str)

    print('Loading data...')
    train_img_pres, validate_img_pres, train_angles_pres, validate_angles_pres, train_img_past, validate_img_past, train_angles_past, validate_angles_past = get_training_data()

    train_inputs = [train_img_pres, train_img_past, train_angles_past]
    validate_inputs = [validate_img_pres, validate_img_past, validate_angles_past]

    print('Training...')
    path = './training'
    if not os.path.exists(path):
        os.makedirs(path)

    start_time = time.time()

    fParams = '{0}/{1}.h5'.format(path, saving)
    saveParams = ModelCheckpoint(fParams, monitor='val_loss', save_best_only=True)

    callbacks = [saveParams]

    model.fit(train_inputs, train_angles_pres, verbose=1,
              validation_data=(validate_inputs, validate_angles_pres),
              nb_epoch=100, batch_size=512, callbacks=callbacks)

    end_time = time.time()

    print('training time: %.2f' % (end_time - start_time))


def main():
    train()

if __name__ == "__main__":
    main()
