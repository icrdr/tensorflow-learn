# %%
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

DATASET_DIR = 'D:\\Dataset\\PetImages'
CATEGORIES = ['Cat', 'Dog']
IMG_SIZE = 50

# %%
train_data = []
for category in CATEGORIES:
    path = os.path.join(DATASET_DIR, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_array = cv2.imread(img_path)
        # cv imread img as bgr as default,
        # so it has to be converted into rgb.
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        train_data.append([img_array, CATEGORIES.index(category)])

# %%
random.shuffle(train_data)


# %%
for sample in train_data[:10]:
    print(sample[1])
    plt.axis("off")
    plt.imshow(sample[0])
    plt.show()


# %%
X = []
Y = []
for img, label in train_data:
    X.append(img)
    Y.append(label)

X = np.array(X)
Y = np.array(Y)
X.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

plt.axis("off")
plt.imshow(X[0])
plt.show()
print(X.shape)


# %%
with open('catdog\\saves\\X.pickle', 'wb') as pickle_f:
    pickle.dump(X, pickle_f)
with open('catdog\\saves\\Y.pickle', 'wb') as pickle_f:
    pickle.dump(Y, pickle_f)

# %%
