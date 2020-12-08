from preprocess import get_data
import os
import sys
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

class YelpClassifier(tf.keras.Model):
    def __init__(self):
        super(YelpClassifier, self).__init__()

        self.batch_size = 100
        self.num_classes = 5

        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.F1 = tf.Variable(tf.random.truncated_normal([5, 5, 3, 16], stddev=.1, dtype=tf.float32))
        self.b1_F = tf.Variable(tf.random.truncated_normal([16], stddev=.1, dtype=tf.float32))
        self.F2 = tf.Variable(tf.random.truncated_normal([5, 5, 16, 20], stddev=.1, dtype=tf.float32))
        self.b2_F = tf.Variable(tf.random.truncated_normal([20], stddev=.1, dtype=tf.float32))
        self.F3 = tf.Variable(tf.random.truncated_normal([3, 3, 20, 20], stddev=.1, dtype=tf.float32))
        self.b3_F = tf.Variable(tf.random.truncated_normal([20], stddev=.1, dtype=tf.float32))

        self.hidden_size = 80

        self.W1 = tf.Variable(tf.random.normal([self.hidden_size, self.hidden_size], stddev=.1, dtype=tf.float32))
        self.b1_W = tf.Variable(tf.random.normal([self.hidden_size], stddev=.1, dtype=tf.float32))
        self.W2 = tf.Variable(tf.random.normal([self.hidden_size, self.hidden_size], stddev=.1, dtype=tf.float32))
        self.b2_W = tf.Variable(tf.random.normal([self.hidden_size], stddev=.1, dtype=tf.float32))
        self.W3 = tf.Variable(tf.random.normal([self.hidden_size, self.num_classes], stddev=.1, dtype=tf.float32))
        self.b3_W = tf.Variable(tf.random.normal([self.num_classes], stddev=.1, dtype=tf.float32))

    def call(self, inputs):
        conv_1 = tf.nn.conv2d(inputs, filters=self.F1, strides=[1, 2, 2, 1], padding="SAME")
        conv_1 = tf.nn.bias_add(conv_1, self.b1_F)
        mean, variance = tf.nn.moments(conv_1, axes=[0, 1, 2])
        norm_1 = tf.nn.batch_normalization(conv_1, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-5)
        relu_1 = tf.nn.relu(norm_1)
        max_pool_1 = tf.nn.max_pool(relu_1, ksize=3, strides=2, padding="SAME")

        conv_2 = tf.nn.conv2d(max_pool_1, filters=self.F2, strides=[1, 2, 2, 1], padding="SAME")
        conv_2 = tf.nn.bias_add(conv_2, self.b2_F)
        mean, variance = tf.nn.moments(conv_2, axes=[0, 1, 2])
        norm_2 = tf.nn.batch_normalization(conv_2, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-5)
        relu_2 = tf.nn.relu(norm_2)
        max_pool_2 = tf.nn.max_pool(relu_2, ksize=2, strides=2, padding="SAME")

        conv_3 = tf.nn.conv2d(max_pool_2, filters=self.F3, strides=[1, 1, 1, 1], padding="SAME")
        conv_3 = tf.nn.bias_add(conv_3, self.b3_F)
        mean, variance = tf.nn.moments(conv_2, axes=[0, 1, 2])
        norm_3 = tf.nn.batch_normalization(conv_3, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-5)
        relu_3 = tf.nn.relu(norm_3)
        relu_3 = tf.reshape(relu_3, (len(inputs), -1))  # reshape before linear layers

        dense_1 = tf.nn.relu(tf.nn.dropout(tf.matmul(relu_3, self.W1) + self.b1_W, rate=0.3))
        dense_2 = tf.nn.relu(tf.nn.dropout(tf.matmul(dense_1, self.W2) + self.b2_W, rate=0.3))
        dense_3 = tf.matmul(dense_2, self.W3) + self.b3_W

        return dense_3

    def loss(self, logits, labels):
        softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))
        return softmax_loss

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    losses = []
    for i in range(0, len(train_inputs), model.batch_size):
        if i + model.batch_size > len(train_inputs):    # remaining batch is too small
            break

        inputs_batch = train_inputs[i:i + model.batch_size]
        labels_batch = train_labels[i:i + model.batch_size]

        with tf.GradientTape() as tape:
            logits = model.call(inputs_batch)
            loss = model.loss(logits, labels_batch)

        losses.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return np.average(np.array(losses))

def test(model, test_inputs, test_labels):
    accuracies = []
    for i in range(0, len(test_inputs), model.batch_size):
        if i + model.batch_size > len(test_inputs):    # remaining batch is too small
            break

        inputs_batch = test_inputs[i:i + model.batch_size]
        labels_batch = test_labels[i:i + model.batch_size]

        logits = model.call(inputs_batch)
        accuracies.append(model.accuracy(logits, labels_batch))
    
    return np.average(np.array(accuracies))

def main():
    # Return the training and testing data and labels from get_data
    train_data, train_labels, test_data, test_labels = get_data("data/json/menu.json", "../yelp-data/photos", size=[32, 32], gan=False, test_one_hot=True)

    # Instantiate model
    model = YelpClassifier()
    
    # Train model
    average_loss = train(model, train_data, train_labels)
    print("Average loss:", average_loss)

    # Test model
    average_accuracy = test(model, test_data, test_labels)
    print("Average accuracy:", average_accuracy)

if __name__ == '__main__':
    main()
