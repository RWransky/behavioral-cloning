'''
Deep Highway Network

Adopted from @awjuliani by @MWransky

'''
import tensorflow as tf
import tensorflow.contrib.slim as slim

num_labels = 99
reg_parameter = 0.001
learn_rate = 0.001
# total layers need to be divisible by 5
total_layers = 5
units_between_stride = int(total_layers / 5)


class Network():
    def __init__(self):
        # The network recieves a batch of images
        self.input_layer = tf.placeholder(shape=[None, 20, 80, 3], dtype=tf.float32, name='input')
        self.angles = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='output')
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
            normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            weights_regularizer=slim.l2_regularizer(reg_parameter)):
            # initial layer fed with batch images
            self.layer = slim.conv2d(self.input_layer, 64, [3, 3],
                scope='conv_'+str(0))
            # build out the highway net units
            for i in range(5):
                for j in range(units_between_stride):
                    self.layer = highwayUnit(self.layer, j+(i*units_between_stride))
                self.layer = slim.conv2d(self.layer, 64, [3, 3],
                    scope='conv_s_'+str(i))
            # extract transition layer
            # self.top = slim.conv2d(self.layer, num_labels, [3, 3],
                # normalizer_fn=slim.batch_norm, activation_fn=None, scope='conv_top')
            self.top = slim.fully_connected(slim.layers.flatten(self.layer), 1,
                activation_fn=None, scope='fully_connected_top')
        # generate softmax probabilities
        self.probs = tf.nn.softmax(self.top)
        # calculate reduce mean loss function
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.angles, self.probs))))
        # optimizer
        self.trainer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        # minimization
        self.update = self.trainer.minimize(self.loss)


def highwayUnit(input_layer, i):
    with tf.variable_scope("highway_unit"+str(i)):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
            H = slim.conv2d(input_layer, 64, [3, 3])
            # Push the network to use the skip connection via a negative init
            T = slim.conv2d(input_layer, 64, [3, 3],
                biases_initializer=tf.constant_initializer(-1.0),
                activation_fn=tf.nn.sigmoid)
            output = H*T + input_layer*(1.0-T)
            return output
