from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Inputs
x = tf.placeholder(tf.float32,[None, 784])
# Weights
W = tf.Variable(tf.zeros([784,10]))
# Bias
b = tf.Variable(tf.zeros([10]))

#Softwmax of the output
y = tf.nn.softmax(tf.matmul(x,W)+b)


#Cross-entropy function that minimizes the error
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

