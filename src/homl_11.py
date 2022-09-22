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