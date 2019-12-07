# %%
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
import h5py

# %%
model = models.load_model('u-net\\models\\unet')

with h5py.File('u-net\\saves\\ready.hdf5', 'r') as f:
    x_train = f["x_train"][:]
    y_train = f["y_train"][:]

x_train = x_train/255
y_train = y_train.astype('float64')
print(x_train.dtype)
print(y_train.dtype)

# %%
# predict test
start = 10
num = 10

preds = model.predict(x_train[start:(start+num)])
pred_dinary = np.array(preds > 0.4)
x_test = x_train[start:(start+num)]
y_test = y_train[start:(start+num)]

fig, axs = plt.subplots(num, 3, figsize=(15, num*5))
for i in range(num):
    axs[i, 0].imshow(x_test[i], cmap='hot')
    axs[i, 1].imshow(y_test[i][:, :, 0], cmap='gray')
    axs[i, 2].imshow(pred_dinary[i][:, :, 0], cmap='gray')

plt.show()
