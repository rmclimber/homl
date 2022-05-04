'''
Author: Rick Morris
Code from: Hands-On Machine Learning, Chapter 10

Other credits to: Aurélien Géron (author of HOML)
'''

import tensorflow as tf
import tensorflow.keras as keras

# method definitions
def divide_by_255(data):
    return data / 255.0

## SET UP DATA
# import the fashion data
fashion = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (Xtest, ytest) = fashion.load_data()

X_train_full_shape = X_train_full.shape
img_dims = [X_train_full_shape[1], X_train_full_shape[2]]
print("Image dimensions: {}".format(img_dims))

# scale pixel intensities and set aside first 5000 samples as validation set
X_val, X_train = divide_by_255(X_train_full[:5000]), divide_by_255(X_train_full[5000:])
y_val, y_train = divide_by_255(y_train_full[:5000]), divide_by_255(y_train_full[5000:])


class_Names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## PREPARE MODEL
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=img_dims))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

print(model.summary())