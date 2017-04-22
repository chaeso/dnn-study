# Lab 11 MNIST and Deep learning CNN
# https://www.tensorflow.org/tutorials/layers
import tensorflow as tf
import numpy as np
import os

import loader as input_data
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# Check out https://www.tensorflow.org/get_started/mnist/beginners for\
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100
NUM_MODELS = 3
CP_PATH = './model2.ckpt'

#20 => 0.9951


class Model:

    def __init__(self, sess, name, dropout_rate=0.3, mask_size=3):
        self.sess = sess
        self.name = name
        self._build_net(dropout_rate, mask_size)

    def _build_net(self, dropout_rate, mask_size):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])

            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional Layer #1
            sz_filter = 20
            conv1 = tf.layers.conv2d(inputs=X_img, filters=sz_filter, kernel_size=[mask_size, mask_size], padding="SAME", activation=None)
            conv1 = tf.nn.elu(tf.layers.batch_normalization(conv1))

            conv1 = tf.layers.conv2d(inputs=conv1, filters=sz_filter, kernel_size=[mask_size, mask_size], padding="SAME", activation=None)
            conv1 = tf.nn.elu(tf.layers.batch_normalization(conv1))

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=pool1, filters=sz_filter*3, kernel_size=[mask_size, mask_size], padding="SAME", activation=None)
            conv2 = tf.nn.elu(tf.layers.batch_normalization(conv2))

            conv2 = tf.layers.conv2d(inputs=conv2, filters=sz_filter*3, kernel_size=[mask_size, mask_size], padding="SAME", activation=None)
            conv2 = tf.nn.elu(tf.layers.batch_normalization(conv2))

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            # Convolutional Layer #2 and Pooling Layer #2
            conv3 = tf.layers.conv2d(inputs=pool2, filters=sz_filter*5, kernel_size=[mask_size, mask_size], padding="same", activation=None)
            conv3 = tf.nn.elu(tf.layers.batch_normalization(conv3))

            conv3 = tf.layers.conv2d(inputs=conv3, filters=sz_filter*5, kernel_size=[mask_size, mask_size], padding="same", activation=None)
            conv3 = tf.nn.elu(tf.layers.batch_normalization(conv3))
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="same", strides=2)


            # Convolutional Layer #3 and Pooling Layer #2
            conv4 = tf.layers.conv2d(inputs=pool3, filters=sz_filter * 10, kernel_size=[mask_size, mask_size],
                                     padding="same", activation=None)
            conv4 = tf.nn.elu(tf.layers.batch_normalization(conv4))

            conv4 = tf.layers.conv2d(inputs=conv4, filters=sz_filter * 10, kernel_size=[mask_size, mask_size],
                                     padding="same", activation=None)
            conv4 = tf.nn.elu(tf.layers.batch_normalization(conv4))
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2],
                                            padding="same", strides=2)

            # Dense Layer with Relu
            flat = tf.reshape(pool4, [-1, sz_filter * 10 * 2 * 2])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=None)
            dense4 = tf.nn.elu(tf.layers.batch_normalization(dense4))

            dense5 = tf.layers.dense(inputs=dense4, units=625, activation=None)
            dense5 = tf.nn.elu(tf.layers.batch_normalization(dense5))

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dense5, units=10)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

# initialize
sess = tf.Session()

models = []
num_models = NUM_MODELS
for m in range(num_models):
    models.append(Model(sess, "model" + str(m), mask_size=3))

last_epoch = tf.Variable(0, name='last_epoch')

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(".")

if checkpoint and checkpoint.model_checkpoint_path:
    try:
        saver.restore(sess, CP_PATH)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    except Exception as e:
        print(str(e))
        print("Error on loading old network weights")


print('Learning Started!')

start_from = sess.run(last_epoch)
print("start_from, ", start_from)

# train my model
for epoch in range(start_from, training_epochs):
    print("epoch = ", epoch)
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        if i % 100 == 0:
            print("batch ", i, "/ ", total_batch, "batch_size", batch_size)
        batch_xs, batch_ys = mnist.train.next_distorted_batch(batch_size)

        # train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
    print(sess.run(last_epoch.assign(epoch + 1)))
    saver.save(sess, CP_PATH)

print('Learning Finished!')

# Test model and check accuracy
test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)
for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(
        mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(
    tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(
    tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))
