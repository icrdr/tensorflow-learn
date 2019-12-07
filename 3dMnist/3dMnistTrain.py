# %%
import h5py
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure
from tensorflow.keras import utils, layers, callbacks, models, optimizers

DATASET_DIR = "D:\\Dataset\\3DMNIST_data"

# %%
# to list all th datasets in h5 file
with h5py.File(DATASET_DIR+'\\full_dataset_vectors.h5', 'r') as f:
    print(list(f.keys()))

# %%
with h5py.File(DATASET_DIR+'\\full_dataset_vectors.h5', 'r') as f:
    x_train = f["X_train"][:]
    y_train = f["y_train"][:]
    x_test = f["X_test"][:]
    y_test = f["y_test"][:]
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

# %%
# reshape it to a 3d volume like dataset
x_train = x_train.reshape((x_train.shape[0], 16, 16, 16, 1))
x_test = x_test.reshape((x_test.shape[0], 16, 16, 16, 1))

print(x_train.shape)
print(x_test.shape)

# convert target variable into one-hot
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

print(y_train.shape)
print(y_test.shape)

# %%


def show3d(vols):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    vols = vols.reshape(vols.shape[0], vols.shape[1], vols.shape[2])
    vols = vols.transpose(1, 2, 0)
    verts, faces, normals, values = measure.marching_cubes_lewiner(vols, 0.2)

    mesh = Poly3DCollection(
        verts[faces], facecolor="gray",
        edgecolor="black", alpha=0.5, linewidth=0.1
    )
    ax.add_collection3d(mesh)

    ax.set_xlim(0, vols.shape[0])
    ax.set_ylim(0, vols.shape[1])
    ax.set_zlim(0, vols.shape[2])


i = 10
print(y_train[i])
show3d(x_train[i])


# %%
# input layer
input_layer = layers.Input((16, 16, 16, 1))

# convolutional layers
conv_layer1 = layers.Conv3D(8, (3, 3, 3), activation='relu')(input_layer)
conv_layer2 = layers.Conv3D(16, (3, 3, 3), activation='relu')(conv_layer1)

# add max pooling to obtain the most imformatic features
pooling_layer1 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

conv_layer3 = layers.Conv3D(32, (3, 3, 3), activation='relu')(pooling_layer1)
conv_layer4 = layers.Conv3D(64, (3, 3, 3), activation='relu')(conv_layer3)
pooling_layer2 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

# perform batch normalization on the convolution outputs
# before feeding it to MLP architecture
pooling_layer2 = layers.BatchNormalization()(pooling_layer2)
flatten_layer = layers.Flatten()(pooling_layer2)

# create an MLP architecture with dense layers : 4096 -> 512 -> 10
# add dropouts to avoid overfitting / perform regularization
dense_layer1 = layers.Dense(2048, activation='relu')(flatten_layer)
dense_layer1 = layers.Dropout(0.4)(dense_layer1)
dense_layer2 = layers.Dense(512, activation='relu')(dense_layer1)
dense_layer2 = layers.Dropout(0.4)(dense_layer2)
output_layer = layers.Dense(10, activation='softmax')(dense_layer2)

model = models.Model(inputs=input_layer, outputs=output_layer)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adadelta(learning_rate=0.1),
    metrics=['acc']
)
NAME = "3d-{}".format(datetime.now().strftime("%H%M%S"))
tensorboard = callbacks.TensorBoard(log_dir='3dMnist\\logs\\%s' % NAME)
model.fit(
    x=x_train,
    y=y_train,
    batch_size=128,
    epochs=8,
    validation_split=0.2,
    callbacks=[tensorboard]
)


# %%
model.save('3dMnist\\models\\3dmnist')

# %%
# predict test
pred = model.predict(x_test[:10])


def convert_to_number(num):
    return np.argmax(num, axis=1)


print(convert_to_number(y_test[:10]))
print(convert_to_number(pred))

# %%
