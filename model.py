from preprocess import get_data, preprocess_images

import numpy as np
import tensorflow as tf

class YelpClassifier(tf.keras.Model):
    def __init__(self, train_size, test_size):
        super(YelpClassifier, self).__init__()

        self.batch_size = 100
        self.num_epoch = 1
        self.num_classes = 9    # ratings of 1.0, 1.5 ... 5.0 stars
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
    train_data, train_labels, test_data, test_labels = get_data("../../Downloads/yelp_dataset/yelp_academic_dataset_business.json", "../../Downloads/yelp_photos-5/photos.json", "../../Downloads/yelp_photos-5/photos", "food")

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
