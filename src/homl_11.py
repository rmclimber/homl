'''
Author: Rick Morris
Code from: Hands-On Machine Learning, Chapter 11

Other credits to: Aurélien Géron (author of HOML)
'''

import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# He initialization, pp. 333-335
# layer can be initialized by setting kernel_initializer
keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal')
he_avg_init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg',
                                                 distributiono='uniform')
keras.layers.Dense(10, activation='sigmoid', kernel_initializer=he_avg_init)

# Batch Normalization
model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(300, activationn='elu', kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(100, activation='elu', kernel_constraint='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation="softmax")
    ]
)

## TRANSFER LEARNING
# transfer learning HOML 347-8
model_A = keras.models.load_model('my_model_A.h5')
model_B_on_A = keras.models.Sequential(model_A.layers[:-1]) # will retrain A
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

# to avoid retraining A
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
# freeze weights
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

## REGULARIZATION
# l1 to have a sparse model; l2 to constrain weights
layer = keras.layers.Dense(100, activation='elu',
                           kernel_initializer='he_normal',
                           kernel_regularizer=keras.regularizers.l2(0.01))
# can use functools.partial to create a wrapper allowing for somem default argument-passing