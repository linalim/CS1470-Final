from preprocess import get_data

import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

def save_model(model, weights_filepath, model_filepath):
    """
    Saves the model weights and architecture out as a h5 and JSON.
    :param weights_filepath: path to where weights will be saved
    :param model_filepath: path to where model will be saved
    """
    model.save_weights(weights_filepath)
    model_json = model.to_json()
    with open(model_filepath, "w") as json_file:
        json_file.write(model_json)
 
# Using ResNet architecture initialized with ImageNet weights and default fully connected layer removed
base_model = ResNet50V2(include_top=False, weights='imagenet')

# Adding custom layers at the end of the network
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(5, activation='softmax')(x)     # star rating of 1 - 5

# Creating a trainable model
model = Model(inputs=base_model.input, outputs=predictions)
 
# Freezing the base_model's layers
# for layer in base_model.layers:
#     layer.trainable = False
 
# Compiling the model
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics = ['accuracy'])

# Return the training and testing data and labels from get_data
train_data, train_labels, test_data, test_labels = get_data("data/json/drink.json", "../yelp-data/photos", size=[75, 75], gan=False, test_one_hot=True)

# Training the model!
model.fit(train_data, train_labels, batch_size=100, epochs=25, verbose=1)
 
# Saving the weights and model architecture
# save_model(model, "resnet_weights.h5", "resnet_model.json")
 
# Evaluate with the test set
test_result = model.evaluate(test_data, test_labels)
print("Accuracy:", test_result[1], " Loss:", test_result[0])
