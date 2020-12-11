import os

import json
from matplotlib.image import imread
import tensorflow as tf
import numpy as np
from progressbar import ProgressBar

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

def get_data(image_json_filepath, image_filepath, size, test_one_hot, gan, test_fraction=0.2):
    """
    Reads image JSON file and images to return data for training and testing.
    :param image_json_filepath: path to JSON file of desired dataset (e.g. food.json)
    :param image_filepath: path to directory with images
    :param size: size [new_height, new_width] to resize each image in image batch to
    :param test_one_hot: True if test labels also need to be one-hot, False otherwise
    :param gan: True if called from GAN, False otherwise
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
    pbar = ProgressBar(maxval=len(images))

    if gan==False:
        for i in pbar(images):
            if i['business_stars'] % 1 == 0:
                # Add business_stars to labels
                labels.append(int(i['business_stars']) - 1)  # label 0 = star 1
                # Read image as numpy array
                img = tf.convert_to_tensor(imread(image_filepath + '/' + i['photo_id'] + '.jpg'), dtype=tf.float32)
                img = tf.image.resize(img, size)
                data.append(img)
    else: # if we're preprocessing for gan, only five-stars
        for i in pbar(images):
            if i['business_stars'] == 5:
                # Add business_stars to labels
                labels.append(int(i['business_stars']) - 1)  # label 0 = star 1
                # Read image as numpy array
                img = tf.convert_to_tensor(imread(image_filepath + '/' + i['photo_id'] + '.jpg'), dtype=tf.float32)
                img = tf.image.resize(img, size)
                data.append(img)


    data = tf.convert_to_tensor(data)
    print("Collected", len(data), "images based on criteria")

    # Split for training and testing
    split_index = int(len(data) * (1 - test_fraction))
    train_data = data[:split_index]
    test_data = data[split_index:]
    train_labels = tf.one_hot(labels[:split_index], 5)
    if test_one_hot:
        test_labels = tf.one_hot(labels[split_index:], 5)
    else:
        test_labels = tf.convert_to_tensor(labels[split_index:])

    print("train_data:", len(train_data))
    print("test_data:", len(test_data))

    return train_data, train_labels, test_data, test_labels

def clean_up_photos():
    """
    Deletes photos that will be unused from photos directory.
    """
    used_photos = set()
    files = [
        "data/json/drink.json",
        "data/json/food.json",
        "data/json/inside.json",
        "data/json/menu.json",
        "data/json/outside.json"
    ]
    for f in files:
        data = read_from_json_file(f)
        for d in data:
            if d['business_stars'] % 1 == 0:
                used_photos.add(d['photo_id'])

    for photo in pbar(os.listdir("../yelp-data/photos")):
        # print(photo)
        if photo.split(".")[0] not in used_photos:
            os.remove("../yelp-data/photos/" + photo)
