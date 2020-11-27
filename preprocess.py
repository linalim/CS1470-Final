import json
from matplotlib.image import imread
import tensorflow as tf
import numpy as np
from progressbar import ProgressBar
pbar = ProgressBar()

def read_from_json_file(filepath):
    """
    Reads JSON file and returns it.
    :param filepath: path to JSON file
    :return: a list of dictionaries (one dict for each JSON object)
    """
    with open(filepath) as infile:
        data = json.load(infile)
    print("Read in", len(data), "objects from", filepath)
    return data

def get_data(image_json_filepath, image_filepath, size=[32, 32], test_fraction=0.2):
    """
    Reads image JSON file and images to return data for training and testing.
    :param image_json_filepath: path to JSON file of desired dataset (e.g. food.json)
    :param image_filepath: path to directory with images
    :param size: size [new_height, new_width] to resize each image in image batch to
    :param test_fraction: fraction of data for testing
    :return: training and testing data and labels
    """
    print("Preprocessing.")

    # Read in image JSON
    images = read_from_json_file(image_json_filepath)

    # Categorize images by label
    data = []
    labels = []
    print("Reading in images.")
    for i in pbar(images):
        if i['business_stars'] % 1 == 0:
            # Add business_stars to labels
            labels.append(int(i['business_stars']) - 1)  # label 0 = star 1
            # Read image as numpy array
            img = tf.convert_to_tensor(imread(image_filepath + '/' + i['photo_id'] + '.jpg'), dtype=tf.float32)
            img = tf.image.resize(img, size)
            data.append(img)

    data = tf.convert_to_tensor(data)
    labels = tf.one_hot(labels, 5)

    print("Collected", len(data), "images based on criteria")

    # Split for training and testing
    split_index = int(len(data) * (1 - test_fraction))
    train_data = data[:split_index]
    test_data = data[split_index:]
    train_labels = labels[:split_index]
    test_labels = labels[split_index:]

    print("train_data:", len(train_data))
    print("test_data:", len(test_data))

    return train_data, train_labels, test_data, test_labels
