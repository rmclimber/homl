'''
Author: Rick Morris
Code from: Hands-On Machine Learning, Chapter 14

Other credits to: Aurélien Géron (author of HOML)
'''


import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# A minibatch is typically represented as a 4D tensor: (batch size, height, width, channels)
from sklearn.datasets import load_sample_image

# load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])
batch_size, height, width, channels = images.shape


# create two filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1 # vertical
filters[3, :, :, 1] = 1 # horizontal

outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")

plt.imshow(outputs[0, :, :, 1], cmap="gray") # plot image1 second feature map
plt.show()

# implementing max pooling output (460)
output = tf.nn.max_pool(images,
                        ksize=(1, 1, 1, 3),
                        strides=(1, 1, 1, 3),
                        padding='valid')
# to use max pooling as a layer
depth_pool = keras.layers.Lambda(
    lambda X: tf.nn.max_pool(X,
                        ksize=(1, 1, 1, 3),
                        strides=(1, 1, 1, 3),
                        padding='valid')
)

# can also do global pooling, output a single number per feature map per instance
# CNN architectures typically get smaller by deeper (more convolutional layers)
# CNN to tackle MNIST
model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation='relu', padding='same',
                        input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activaiton='relu', padding='same'),
    keras.layers.Conv2D(128, 3, activaiton='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activaiton='relu', padding='same'),
    keras.layers.Conv2D(256, 3, activaiton='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(.5),
    keras.layers.Dense(10, activation='softmax')

])