# %%
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imread_collection
from skimage.transform import resize
import os
import h5py

DATASET_DIR = "D:\\Dataset\\nuke"

HEIGHT = 128
WIDTH = 128

# prepare the data
# %%


def generate_data(dir):
    file_dir = os.path.join(DATASET_DIR, dir)
    cases = os.listdir(file_dir)
    images = np.zeros((len(cases), HEIGHT, WIDTH, 3), dtype=np.uint8)
    labels = np.zeros((len(cases), HEIGHT, WIDTH, 1), dtype=np.bool)

    print('Getting and resizing TRAIN images and masks ... ')

    for i, case in tqdm(enumerate(cases), total=len(cases)):
        # get and resize image
        img_path = file_dir+"\\{}\\images\\{}.png".format(case, case)
        # only need RGB, discard A channel
        img_array = imread(img_path)[:, :, :3]
        img_array = resize(img_array, (HEIGHT, WIDTH, 3),
                           mode='constant', preserve_range=True)
        images[i] = img_array

        # get and resize mask
        mask_path = file_dir + "\\{}\\masks\\*".format(case)
        mask_arrays = imread_collection(mask_path).concatenate()
        mask_arrays = np.expand_dims(mask_arrays, -1)

        label_array = np.zeros((HEIGHT, WIDTH, 1), dtype=np.bool)
        for j, mask_array in enumerate(mask_arrays):
            mask_array = resize(
                mask_array, (HEIGHT, WIDTH, 1), mode='constant')
            label_array = np.maximum(label_array, mask_array)
        labels[i] = label_array
    return images, labels
    print('Done!')


img_train, label_train = generate_data('stage1_train')

# %%
plt.imshow(img_train[10])
plt.show()

# %%
with h5py.File("u-net\\saves\\ready.hdf5", "w") as f:
    x_train = f.create_dataset("x_train", data=img_train)
    y_train = f.create_dataset("y_train", data=label_train)

# %%
