'''
Author: Rick Morris
Code from: Hands-On Machine Learning, Chapter 12

Other credits to: Aurélien Géron (author of HOML)
'''


import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## learning Tensors
# create a tensor
tensor = tf.constant([[1., 2., 3.], [4., 5., 6.]])

# mean
mu = tf.reduce_mean(tensor)

# transpose
tt = tf.transpose(tensor)

## VARIABLES 382
# tensors are constant so they cannot be modified like a numpy array
var = tf.Variable([[1., 2., 3.], [4., 5., 6.]])

# tensors can also be stored in various TF data structures like sets and lists