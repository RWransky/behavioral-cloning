'''
Leaf Classifier via Deep Highway Networks

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import tensorflow.contrib.slim as slim

from model_helpers import *
from data_helpers import *
from network import *

# Setting the training parameters

# How many experience traces to use for each training step.
batch_size = 32
# Number of training steps
num_steps = 1001
load_model = False
# The path to save our model to.
path = "./weights"


def train():
    # load training and validation data sets
    train_dataset, valid_dataset, train_labels, valid_labels = get_training_data()
    train_labels = np.uint8(train_labels)
    valid_labels = np.uint8(valid_labels)
    print('Size of training dataset is: {} samples'.format(train_dataset.shape[0]))
    print('Size of validation dataset is: {} samples'.format(valid_dataset.shape[0]))
    tf.reset_default_graph()
    mainN = Network()

    init = tf.initialize_all_variables()

    saver = tf.train.Saver(max_to_keep=5)

    # Make list to store losses
    losses = []
    # Make list to store accuracies
    accuracies = []
    # Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        if load_model is True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)

        for step in range(num_steps):
            print('Processing step {}'.format(step))
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size)]
            _, lossA, yP, L0 = sess.run([mainN.update, mainN.loss, mainN.probs, mainN.label_oh],
                feed_dict={mainN.input_layer: batch_data, mainN.label_layer: batch_labels})
            losses.append(lossA)
            accuracies.append(accuracy(L0, yP))
            if (step % 500 == 0):
                print('Minibatch loss at step %d: %f' % (step, lossA))
                print('Minibatch accuracy: %.1f%%' % accuracy(L0, yP))
                # yP, L0 = sess.run([mainN.probs, mainN.label_oh],
                #     feed_dict={mainN.input_layer: valid_dataset, mainN.label_layer: valid_labels})
                # print('Validation accuracy: %.1f%%' % accuracy(yP, L0))
                saver.save(sess, path+'/model-'+str(step)+'.cptk')
                print("Saved Model")
        yP, L0 = sess.run([mainN.probs, mainN.label_oh],
            feed_dict={mainN.input_layer: valid_dataset, mainN.label_layer: valid_labels})
        print('Validation accuracy: %.1f%%' % accuracy(L0, yP))
        saver.save(sess, path+'/model-'+str(step)+'.cptk')
        print("Saved Model")
        plt.figure(1)
        plt.title('Training Loss')
        plt.plot(range(len(losses)), losses)
        plt.figure(2)
        plt.title('Training Accuracies')
        plt.plot(range(len(accuracies)), accuracies)
        plt.show()


def main():
    train()


if __name__ == "__main__":
    main()
