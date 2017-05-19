import tensorflow as tf
import random
import matplotlib.pyplot as plt

from mnist import MNIST
from plot import display_digit, display_mult_flat 

mnist = MNIST()

session = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Weight and bias
b = tf.Variable(tf.zeros([10]))
W = tf.Variable(tf.zeros([784,10]))

#Softmax
y = tf.nn.softmax(tf.matmul(x,W) +b)

#Cross-entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#Training and testing data
x_train, y_train = mnist.train_size(2500)
x_test, y_test = mnist.test_size(10000)

LEARNING_RATE = 0.1
TRAIN_STEPS = 2500

#Run session
init = tf.global_variables_initializer()
session.run(init)

training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(TRAIN_STEPS+1):
	session.run(training, feed_dict={x: x_train, y_: y_train})
	if i%100 == 0:
		print('Training Step:' + str(i) + '  Accuracy =  ' + str(session.run(accuracy, feed_dict={x: x_test, y_: y_test})) + '  Loss = ' + str(session.run(cross_entropy, {x: x_train, y_: y_train})))


for i in range(10):
    plt.subplot(2, 5, i+1)
    weight = session.run(W)[:,i]
    plt.title(i)
    plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

plt.show()

