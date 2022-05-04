'''
Author: Rick Morris
Code from: Hands-On Machine Learning, Chapter 10

Other credits to: Aurélien Géron (author of HOML)
'''

import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
y_val, y_train = y_train_full[:5000], y_train_full[5000:]
print(X_train.shape)


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## PREPARE MODEL
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=img_dims))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

print(model.summary())
print(model.layers)
print(model.layers[1].get_weights())

## COMPILE MODEL
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

## TRAIN MODEL
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_val, y_val))

# if not satisfied with performance, can all fit() again

## EXAMINE history
print(history.params)
print(history.epoch)

## PLOT HISTORY
plt.close()
pd.DataFrame(history.history).plot(figsize=(0, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set vertical range to [0-1]
plt.show()
plt.close()

## EVALUATE MODEL
model.evaluate(Xtest, ytest)

## MAKE PREDICTIONS
X_new = Xtest[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = model.predict_classes(X_new)
print(y_pred)
print(np.array(class_names)[y_pred])
y_new = ytest[:3]
print(y_new)