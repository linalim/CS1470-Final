from preprocess import get_data
import math
import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

class Gan(tf.keras.Model):
    def __init__(self):
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.EPOCHS = 50000
        self.noise_dim = 100
        self.num_examples_to_generate = 16
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        self.BUFFER_SIZE = 60000
        self.BATCH_SIZE = 256
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((8, 8, 256)))
        assert model.output_shape == (None, 8, 8, 256) # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 32, 32, 3)

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[32, 32, 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(model, images, generator, discriminator):
    noise = tf.random.normal([model.BATCH_SIZE, model.noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = model.generator_loss(fake_output)
        disc_loss = model.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    model.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    model.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(model, dataset, epochs, generator, discriminator, checkpoint):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(model, image_batch, generator, discriminator)

        # Save the model every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            checkpoint.save(file_prefix = model.checkpoint_prefix)
            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                    epoch + 1,
                                    model.seed)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                            epochs,
                            model.seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig('./output/image_at_epoch_{:04d}.png'.format(epoch))
    plt.clf()
    # plt.show()

def display_image(epoch_no):
    return PIL.Image.open('./output/image_at_epoch_{:04d}.png'.format(epoch_no))
    
def main():
    # Load data
    train_images, train_labels, test_data, test_labels = get_data("data/json/food.json", "../yelp-data/photos", size=[32, 32], gan=True, test_one_hot=True)

    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

    gan = Gan()
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(gan.BUFFER_SIZE).batch(gan.BATCH_SIZE)


    # Instantiate generator model
    generator = gan.make_generator_model()

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    plt.imshow(generated_image[0, :, :, 0])

    # Instantiate discriminator model
    discriminator = gan.make_discriminator_model()
    decision = discriminator(generated_image)
    print("decision :", decision) # positive for real images, negative for fake images

    checkpoint = tf.train.Checkpoint(generator_optimizer=gan.generator_optimizer,
                                        discriminator_optimizer=gan.discriminator_optimizer,
                                        generator=generator,
                                        discriminator=discriminator)

    # Train
    fig = plt.figure(figsize=(4,4))
    train(gan, train_dataset, gan.EPOCHS, generator, discriminator, checkpoint)

    checkpoint.restore(tf.train.latest_checkpoint(gan.checkpoint_dir))

    display_image(gan.EPOCHS)


if __name__ == '__main__':
    main()