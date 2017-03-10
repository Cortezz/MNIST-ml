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

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

#training the model
for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	session.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	


print(session.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
