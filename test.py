from preprocess import get_data

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

def load_model(model_filepath, weights_filepath):
    """
    Loads in model and weights.
    :param model_filepath: path to where model will be saved
    :param weights_filepath: path to where weights will be saved
    """
    # Load in model
    with open(model_filepath, 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

    # Load in weights
    loaded_model.load_weights(weights_filepath)
    return loaded_model

model = load_model('resnet50_model.json', 'weights/menu.h5')
_, _, test_data, test_labels = get_data("data/json/menu.json", "../yelp-data/photos", size=[32, 32], test_one_hot=False)

predictions = tf.argmax(model.predict(test_data), 1)
print("Accuracy on test set:", tf.math.count_nonzero(tf.math.equal(predictions, test_labels)) / len(test_labels))
