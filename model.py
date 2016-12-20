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
batch_size = 8
# Number of training steps
num_steps = 251
load_model = False
# The path to save our model to.
path = "./weights"


def train():
    # load training and validation data sets
    train_dataset, valid_dataset, train_angles, valid_angles = get_training_data()

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
            offset = (step * batch_size) % (train_angles.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_angles = train_angles[offset:(offset + batch_size)]
            _, lossA, yP = sess.run([mainN.update, mainN.loss, mainN.probs],
                feed_dict={mainN.input_layer: batch_data, mainN.angles: batch_angles})
            losses.append(lossA)
            accuracies.append(accuracy(batch_angles, yP))
            if (step % 50 == 0):
                print('Minibatch loss at step %d: %f' % (step, lossA))
                print('Minibatch accuracy: %.1f%%' % accuracy(yP, batch_angles))
                yP = sess.run([mainN.probs],
                    feed_dict={mainN.input_layer: valid_dataset, mainN.angles: valid_angles})
                print('Validation accuracy: %.1f%%' % accuracy(yP, valid_angles))
                saver.save(sess, path+'/model-'+str(step)+'.cptk')
                print("Saved Model")
        yP = sess.run([mainN.probs],
            feed_dict={mainN.input_layer: valid_dataset, mainN.angles: valid_angles})
        print('Validation accuracy: %.1f%%' % accuracy(yP, valid_angles))
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
