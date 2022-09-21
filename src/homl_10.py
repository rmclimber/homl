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
history = model.fit(X_train, y_train, epochs=1,
                    validation_data=(X_val, y_val))
mse_teset = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
print(y_test[:3])
print(y_pred)


# This is the Functional API (layers are called like functions and passed their input layer)
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input_)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])

## CREATE WIDE AND DEEP NET 308-309
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
# at this point, a second output could be added if we wanted
model = keras.Model(input=[input_A, input_B], outputs=[output])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

# to use this new model, have to pass it two input matrices
X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_val[:, :5], X_val[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                    validation_data=((X_valid_A, X_valid_B), y_val))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))

# can send subset of features through wide path and other subset through deep path (p. 310)
# primary difference is that we must pass pair of matrices to train(), fit(), predict()
# also need to name the input layers, so that dict can be used to pass inputs

# use cases: finding location of main object (classification = main object, regression = coords)
# use cases: multitask classification: one to identify whether wearing glasses, another to ID whether smiling
# use case: regularization

## SUBCLASSING API
# models can be subclassed using the Sublcassing API
# Keras models can be used just like regular layers
class WideAndDeep(keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        # note that we do not immediately call the new models as functions until here
        input_a, input_b = inputs
        hidden1 = self.hidden1(input_b)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_a, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output


# saving and restoring a model with Sequential or Functional
model = keras.models.Sequential([...])
model.compile([...])
model.fit([...])
model.save([...])
model = keras.models.load_model([...])

# saving and restoring with subclassing: requires  requires load/save_weights (p. 315)

# callbacks
# fit accepts a callbacks argument---keras.callbacks.ModelCheckpoint() can be used to save checkpoints



# can save Sequential/Functional API model with model.save('filename.h5'); load_model('filename.h5') to get it up andn running
# ^ cannot save subclassed model`

# callbacks can be used to save history at various points, by passingn array of callbacks to callbacks parameter

# TENSORBOARD
## TensorBoard can be used to see results of hyperparameter tuning, is discussed 317-320
# done as callback to fit()

## Using scikitlearn for hyperparam tuning

def build_model(n_hidden=1, n_neurons=10, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

# wraps result of build_model into a sklearn regressor
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

keras_reg.fit(X_train, y_train, epochs=100,
              validation_data=(X_val, y_val),
              callbacks=[keras.callbacks.EarlyStopping(patience=10)]) # extra params here go to underlying model
mse_test = keras_reg.score(X_test, y_test)
y_pred = keras_reg.predict(X_new)

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_val, y_val),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

# p. 322 has efficient hyperparameter-tuning libraries to try


# deep neural nets can use exponentiailly fewer neurons to model complex phenomena (323-4)
# usually fine to use same size for each hidden layer
# better to go too big and then use early stopping (etc) to regularize