import json
import cv2
import numpy as np

# Reads JSON file and returns a list of dictionaries (one dict for each JSON object).
# filepath - path to JSON file
def read_from_json_file(filepath):
    data = []
    with open(filepath) as infile:
        for l in infile:
            data.append(json.loads(l))
    print("Read in", len(data), "objects from", filepath)
    return data

# Return training and testing data and labels for given label from business and image JSON.
# business_json_filepath - path to yelp_academic_dataset_business.json
# image_json_filepath - path to photos.json
# image_filepath - path to directory with images
# image_label - label to filter on
# size - size (new_height, new_width) to resize each image in image batch to (ex: [32, 32])
# test_fraction - fraction of data for testing
def get_data(business_json_filepath, image_json_filepath, image_filepath, image_label, size, test_fraction=0.2):
    assert(image_label in ['food', 'menu', 'drink', 'inside', 'outside'])
    # Read in business JSON
    businesses = read_from_json_file(business_json_filepath)

    # Create dictionary of business_id -> stars
    business_ratings = {}
    for b in businesses:
        business_ratings[b['business_id']] = b['stars']

    # Read in image JSON
    images = read_from_json_file(image_json_filepath)

    # Categorize images by label
    data = []
    labels = []
    for i in images:
        # if i['label'] == label and i['business_stars'] % 1 == 0:
        if i['label'] == image_label:
            # Add business_stars to labels
            labels.append(business_ratings[i['business_id']])
            data.append(cv2.imread(image_filepath + i['photo_id'] + '.jpeg', mode='RGB'))
            # data.append(i)

    print("Found", len(data), "images with label", label)

    split_index = int(len(data) * (1 - test_fraction))
    train_data = data[:split_index]
    test_data = data[split_index:]
    train_labels = labels[:split_index]
    test_labels = labels[split_index:]

    print("train_data:", len(train_data))
    print("test_data:", len(test_data))

    return train_data, train_labels, test_data, test_labels

# Preprocesses given batch of images.
# image_batch - batch of images
# size - size (new_height, new_width) to resize each image in image batch to (ex: [32, 32])
def preprocess_images(image_batch, size):
    # image_batch = tf.image.rgb_to_grayscale(image_batch)
    return tf.image.resize(image_batch, size)

# get_data("../../Downloads/yelp_dataset/yelp_academic_dataset_business.json", "../../Downloads/yelp_photos-5/photos.json", "food", test_fraction=0.2)
