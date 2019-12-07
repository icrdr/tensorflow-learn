# %%
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras


# %%
X = pickle.load(open('catdog\\saves\\X.pickle', 'rb'))
Y = pickle.load(open('catdog\\saves\\Y.pickle', 'rb'))

X = X/255.0


# %%
dropout_sets = [0.5]
layer_sizes = [64]
conv_layers = [4]
for dropout_set in dropout_sets:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dropout-{}".format(
                conv_layer, layer_size, dropout_set, int(time.time()))
            tensorboard = TensorBoard(log_dir='catdog\\logs\\%s' % NAME)

            model = keras.models.Sequential()
            model.add(Conv2D(layer_size, (3, 3),
                             input_shape=X.shape[1:], activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for i in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dropout(dropout_set))

            model.add(Dense(1, activation='sigmoid'))

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            model.fit(X, Y, epochs=12, batch_size=32,
                      validation_split=0.3, callbacks=[tensorboard])

# %%
model.save('catdog\\models\\catdog-64x4')


# %%
