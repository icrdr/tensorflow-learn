# %%

from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras import layers, callbacks, models, optimizers, backend
import h5py

DATASET_DIR = "u-net\\saves"

# %%
with h5py.File(DATASET_DIR+'\\ready.hdf5', 'r') as f:
    x_train = f["x_train"][:]
    y_train = f["y_train"][:]

x_train = x_train/255
y_train = y_train.astype('float64')
print(x_train.dtype)
print(y_train.dtype)


# %%
NUM = 10
fig, axs = plt.subplots(NUM, 2, figsize=(10, NUM*5))
for i in range(NUM):
    axs[i, 0].imshow(x_train[i], cmap='hot')
    axs[i, 1].imshow(y_train[i][:, :, 0])

# %%
backend.clear_session()
# input layer
input_layer = layers.Input((128, 128, 3))

# convolutional layers 128 > 128 > 128
conv_1 = layers.Conv2D(32, (3, 3), padding='same',
                       activation='relu')(input_layer)
conv_2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(conv_1)

# max pooling 128 > 64
pooling_1 = layers.MaxPool2D(pool_size=(2, 2))(conv_2)

# convolutional layers 64 > 64 > 64
conv_3 = layers.Conv2D(64, (3, 3), padding='same',
                       activation='relu')(pooling_1)
conv_4 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv_3)

# max pooling 64 > 32
pooling_2 = layers.MaxPool2D(pool_size=(2, 2))(conv_4)

# convolutional layers 32 > 32 > 32
conv_5 = layers.Conv2D(128, (3, 3), padding='same',
                       activation='relu')(pooling_2)
conv_6 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv_5)

# max pooling 32 > 16
pooling_3 = layers.MaxPool2D(pool_size=(2, 2))(conv_6)

# convolutional layers 16 > 16 > 16
conv_7 = layers.Conv2D(256, (3, 3), padding='same',
                       activation='relu')(pooling_3)
conv_8 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(conv_7)

# up convolutional layers 16 > 32
up_conv_1 = layers.Conv2DTranspose(
    128, (2, 2), strides=2, padding='same')(conv_8)
merge_1 = layers.Concatenate()([up_conv_1, conv_6])

# convolutional layers 32 > 32 > 32
conv_9 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(merge_1)
conv_10 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv_9)

# up convolutional layers 32 > 64
up_conv_2 = layers.Conv2DTranspose(
    64, (2, 2), strides=2, padding='same')(conv_10)
merge_2 = layers.Concatenate()([up_conv_2, conv_4])

# convolutional layers 64 > 64 > 64
conv_11 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(merge_2)
conv_12 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv_11)

# up convolutional layers 64 > 128
up_conv_3 = layers.Conv2DTranspose(
    32, (2, 2), strides=2, padding='same')(conv_12)
merge_3 = layers.Concatenate()([up_conv_3, conv_2])

# convolutional layers 128 > 128 > 128
conv_13 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(merge_3)
conv_14 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(conv_13)
drop_1 = layers.Dropout(0.5)(conv_14)

# convolutional layers 128 128
output_layer = layers.Conv2D(1, (1, 1), padding='same')(drop_1)
model = models.Model(inputs=input_layer, outputs=output_layer)

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.001),
    metrics=['acc']
)

NAME = "u-net-{}".format(datetime.now().strftime("%H%M%S"))
tensorboard = callbacks.TensorBoard(log_dir='u-net\\logs\\%s' % NAME)
model.fit(
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=60,
    validation_split=0.3,
    callbacks=[tensorboard]
)

# %%
model.save('u-net\\models\\unet')
# %%
