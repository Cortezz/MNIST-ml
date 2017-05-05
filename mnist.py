from tensorflow.examples.tutorials.mnist import input_data


class MNIST:
    def __init__(self):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    def train_size(self, num):
        print('Total training images in dataset: {}'.format(str(self.mnist.train.images.shape)))
        x_train = self.mnist.train.images[:num,:]
        y_train = self.mnist.train.labels[:num,:]
        print('X - examples loaded: {}'.format(str(x_train.shape)))
        print('Y - examples loaded: {}'.format(str(y_train.shape)))
        return x_train, y_train

    def test_size(self, num):
        print('Total test images in dataset: {}'.format(str(self.mnist.test.images.shape)))
        x_test = self.mnist.test.images[:num,:]
        y_test = self.mnist.test.labels[:num,:]
        print('X - tested : {}'.format(x_test.shape))
        print('Y - tested : '.format(y_test.shape))
