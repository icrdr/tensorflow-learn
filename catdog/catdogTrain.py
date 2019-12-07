# %%
import pickle
from tensorflow.keras import layers, models, callbacks, optimizers
from datetime import datetime

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
            model = models.Sequential()
            model.add(layers.Conv2D(
                layer_size, (3, 3),
                input_shape=X.shape[1:], activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            for i in range(conv_layer-1):
                model.add(layers.Conv2D(layer_size, (3, 3), activation='relu'))
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            model.add(layers.Flatten())
            model.add(layers.Dropout(dropout_set))

            model.add(layers.Dense(1, activation='sigmoid'))

            model.compile(
                loss='binary_crossentropy',
                optimizer=optimizers.Adam(learning_rate=0.1),
                metrics=['acc']
            )
            NAME = "{}-conv-{}-nodes-{}-dropout-{}".format(
                conv_layer, layer_size, dropout_set,
                datetime.now().strftime("%H%M%S")
            )
            tensorboard = callbacks.TensorBoard(
                log_dir='catdog\\logs\\%s' % NAME)
            model.fit(
                x=X,
                y=Y,
                batch_size=32,
                epochs=12,
                validation_split=0.3,
                callbacks=[tensorboard]
            )

# %%
model.save('catdog\\models\\catdog-64x4')


# %%
