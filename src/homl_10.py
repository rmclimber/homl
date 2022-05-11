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
history = model.fit(X_train, y_train, epochs=1,
                    validation_data=(X_val, y_val))

# if not satisfied with performance, can all fit() again

## EXAMINE history
print(history.params)
print(history.epoch)

## PLOT HISTORY
print('PLOTTING')
plt.close()
pd.DataFrame(history.history).plot(figsize=(0, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set vertical range to [0-1]
plt.show()
# plt.close()

## EVALUATE MODEL
print('EVALUATE')
model.evaluate(Xtest, ytest)

## MAKE PREDICTIONS
print('PREDICT')
X_new = Xtest[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

print('PREDICT CLASSES')
y_pred = model.predict_classes(X_new)
print(y_pred)
print(np.array(class_names)[y_pred])
y_new = ytest[:3]
print(y_new)

### Building a regression MLP (p. 307)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(X_train.shape)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='sgd')
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_val, y_val))
mse_teset = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)


