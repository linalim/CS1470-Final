import json
from matplotlib.image import imread
from cv2 import imread, resize
import tensorflow as tf
import numpy as np

def read_from_json_file(filepath):
    """
    Reads JSON file and returns it.
    :param filepath: path to JSON file
    :return: a list of dictionaries (one dict for each JSON object)
    """
    data = []
    with open(filepath) as infile:
        for l in infile:
            data.append(json.loads(l))
    print("Read in", len(data), "objects from", filepath)
    return data

def get_data(business_json_filepath, image_json_filepath, image_filepath, image_label, size=[32, 32], test_fraction=0.2):
    """
    Reads business and image JSON files and images to return data for training and testing.
    :param business_json_filepath: path to yelp_academic_dataset_business.json
    :param image_json_filepath: path to photos.json
    :param image_filepath: path to directory with images
    :param image_label: label to filter on
    :param size: size [new_height, new_width] to resize each image in image batch to
    :param test_fraction: fraction of data for testing
    :return: training and testing data and labels for given label from business and image JSON
    """
    assert(image_label in ['food', 'menu', 'drink', 'inside', 'outside'])
    # Read in business JSON
    businesses = read_from_json_file(business_json_filepath)

    # Create dictionary of business_id -> stars
    business_ratings = {}
    for b in businesses:
        business_ratings[b['business_id']] = b['stars']

    # Read in image JSON
    images = read_from_json_file(image_json_filepath)

    # TO DELETE
    images = images[:1000] # first 1000 for testing purposes

    # Categorize images by label
    data = []
    labels = []
    for i in images:
        # if i['label'] == label and i['business_stars'] % 1 == 0:
        if i['label'] == image_label:
            # Add business_stars to labels
            labels.append(business_ratings[i['business_id']])
            # Read image as numpy array
            data.append(imread(image_filepath + '/' + i['photo_id'] + '.jpg'))

    # Resize images to given size
    # data = tf.image.resize(tf.convert_to_tensor(data), size)
    resized_data = []
    for d in data:
        resized_data.append(resize(d, tuple(size)))
    data = resized_data

    print("Found", len(data), "images with label", labels)

    # Split for training and testing
    split_index = int(len(data) * (1 - test_fraction))
    train_data = data[:split_index]
    test_data = data[split_index:]
    train_labels = labels[:split_index]
    test_labels = labels[split_index:]

    print("train_data:", len(train_data))
    print("test_data:", len(test_data))

    return train_data, train_labels, test_data, test_labels
