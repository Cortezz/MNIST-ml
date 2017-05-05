import tensorflow as tf
import random

from mnist import MNIST
from plot import display_digit

mnist = MNIST()

x_train, y_train = mnist.train_size(55000)
num = random.randint(0, x_train.shape[0])

label = y_train[num].argmax(axis=0)
image = x_train[num].reshape([28, 28])

display_digit(image)
