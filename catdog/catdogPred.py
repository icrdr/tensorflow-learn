# %%
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

IMG_SIZE = 50
DATASETDIR = 'D:\\Dataset\\test\\Dog'


def imgPrepera(path):
    img_array = cv2.imread(os.path.join(DATASETDIR, path))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return np.array(img_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


# %%
model = keras.models.load_model('catdog-64x4')
predictions = model.predict([imgPrepera('images.jpg')])
print(predictions)

# %%
