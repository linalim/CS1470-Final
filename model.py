from preprocess import get_data, preprocess_images

import numpy as np
import tensorflow as tf

class YelpClassifier(tf.keras.Model):
    def __init__(self, train_size, test_size):
        super(YelpClassifier, self).__init__()

        # self.batch_size = 100
        self.num_epoch = 1
        # self.num_classes = 2
        self.dropout_rate = 0.5

        # self.hidden_layer_size1 = 20

        self.conv1 = tf.Variable(tf.random.truncated_normal([5, 5, 3, 5], stddev=0.1))
        self.b1 = tf.Variable(tf.random.truncated_normal([5], stddev=0.1))

        self.conv2 = tf.Variable(tf.random.truncated_normal([5, 5, 5, 10], stddev=0.1))
        self.b2 = tf.Variable(tf.random.truncated_normal([10], stddev=0.1))

        self.linear_b1 = tf.Variable(tf.random.truncated_normal([100], stddev=0.1))
        self.linear_b2 = tf.Variable(tf.random.truncated_normal([1], stddev=0.1))
        self.linear_W1 = tf.Variable(tf.random.truncated_normal([28090, 100], stddev=0.1))
        self.linear_W2 = tf.Variable(tf.random.truncated_normal([100, 1], stddev=0.1))

    def call(self, inputs):

        # CONVOLUTION LAYER 1
        x = tf.nn.conv2d(inputs, self.conv1)
        x = tf.nn.bias_add(x, self.b1)
        # MAX POOLING 1
        x = tf.nn.max_pool(x, ksize=[2,2])
        # RELU 1
        x = tf.nn.relu(x)

        #CONVOLUTION LAYER 2
        x = tf.nn.conv2d(x, self.conv2)
        x = tf.nn.bias_add(x, self.b2)
        # MAX POOLING 2
        x = tf.nn.max_pool(x, ksize=[2,2])
        # RELU 2
        x = tf.nn.relu(x)

        # RESHAPE
        x = tf.reshape(x, [-1, 28090])

        # DENSE LAYER 1
        x = tf.matmul(x, self.linear_W1) + self.linear_b1
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, rate=self.dropout_rate)

        # DENSE LAYER 2
        x = tf.matmul(x, self.linear_W2) + self.linear_b2

        return x

    def loss(self, logits, labels):
        softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1))
        return softmax_loss

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def main():
    # Return the training and testing data from get_data
    train_data, test_data = get_data("../../Downloads/yelp_dataset/yelp_academic_dataset_business.json", "../../Downloads/yelp_photos-5/photos.json", "food", test_fraction=0.2)

    # Instantiate model
    model = YelpClassifier()

    # Train and test for up to 15 epochs.
    epochs = 15
    for i in range(epochs):
        print("Epoch:", i)
        train(model, train_data)
        accuracy = test(model, test_data)
        print("Batch accuracy:", accuracy)
        print()


if __name__ == '__main__':
    main()
