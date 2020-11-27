from preprocess import get_data
import os
import numpy as np
import math
import tensorflow as tf

class YelpClassifier(tf.keras.Model):
    def __init__(self, resnet):
        super(YelpClassifier, self).__init__()

        # self.resnet_conv = resnet(include_top=True, weights='imagenet', pooling='max', classifier_activation='relu')
        # self.resnet_avg = resnet(include_top=True, weights='imagenet', pooling='avg')
        # self.linear = tf.keras.layers.Dense(512)

        self.batch_size = 100
        self.num_epoch = 1
        self.num_classes = 9    # ratings of 1.0, 1.5 ... 5.0 stars
        self.dropout_rate = 0.5
        self.learning_rate = 1e-3
        self.hidden_layer1 = 80
        self.hidden_layer2 = 40
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        self.conv1 = tf.Variable(tf.random.truncated_normal([5, 5, 3, 5], stddev=0.1))
        self.b1 = tf.Variable(tf.random.truncated_normal([5], stddev=0.1))

        self.conv2 = tf.Variable(tf.random.truncated_normal([5, 5, 5, 10], stddev=0.1))
        self.b2 = tf.Variable(tf.random.truncated_normal([10], stddev=0.1))

        self.linear_b1 = tf.Variable(tf.random.truncated_normal([self.hidden_layer2], stddev=0.1))
        self.linear_b2 = tf.Variable(tf.random.truncated_normal([self.num_classes], stddev=0.1))
        self.linear_W1 = tf.Variable(tf.random.truncated_normal([self.hidden_layer1, self.hidden_layer2], stddev=0.1))
        self.linear_W2 = tf.Variable(tf.random.truncated_normal([self.hidden_layer2, self.num_classes], stddev=0.1))

    def call(self, inputs):
        # BASIC NET
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # CONVOLUTION LAYER 1
        x = tf.nn.conv2d(inputs, self.conv1, strides=[1,1,1,1], padding='SAME')
        x = tf.nn.bias_add(x, self.b1)
        # MAX POOLING 1
        x = tf.nn.max_pool(x, ksize=[2,2], strides=[1,1,1,1], padding='SAME')
        # RELU 1
        x = tf.nn.relu(x)

        #CONVOLUTION LAYER 2
        x = tf.nn.conv2d(x, self.conv2, strides=[1,1,1,1], padding='SAME')
        x = tf.nn.bias_add(x, self.b2)
        # MAX POOLING 2
        x = tf.nn.max_pool(x, ksize=[2,2], strides=[1,1,1,1], padding='SAME')
        # RELU 2
        x = tf.nn.relu(x)

        # RESHAPE
        x = tf.reshape(x, [-1, self.hidden_layer1])

        # DENSE LAYER 1
        x = tf.matmul(x, self.linear_W1) + self.linear_b1
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, rate=self.dropout_rate)

        # DENSE LAYER 2
        x = tf.matmul(x, self.linear_W2) + self.linear_b2


        # RESNET
        # x = self.resnet_conv(inputs)
        # x = self.resnet_avg(x)
        # x = tf.reshape(x, [x.shape[0], -1])

        # x = self.linear(x)

        return x

    def loss(self, logits, labels):
        softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1))
        return softmax_loss

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    print("training")
    losses = []
    for i in range(0, len(train_inputs), model.batch_size):
        if i + model.batch_size > len(train_inputs):    # remaining batch is too small
            break

        inputs_batch = train_inputs[i:i + model.batch_size]
        labels_batch = train_labels[i:i + model.batch_size]

        with tf.GradientTape() as tape:
            logits = model.call(inputs_batch)
            print("LOOK HERE")
            print(tf.shape(labels_batch))
            loss = model.loss(logits, labels_batch)

        losses.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return np.average(np.array(losses))

def test(model, test_inputs, test_labels):
    print("testing")
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
    train_data, train_labels, test_data, test_labels = get_data("../yelp-data/yelp_academic_dataset_business.json", "../yelp-data/photos.json", "../yelp-data/photos", "food")

    # Instantiate model
    m = tf.keras.applications.ResNet101V2()
    model = YelpClassifier(m)

    # Train model
    average_loss = train(model, train_data, train_labels)
    print("Average loss:", average_loss)

    # Test model
    average_accuracy = test(model, test_data, test_labels)
    print("Average accuracy:", average_accuracy)

if __name__ == '__main__':
    main()
