from __future__ import division

import os
import numpy as np
import random
from preprocess import *

def get_data(data_df, data_folder, labels, batch_size=32, shuffle=True,
             augmentation=True, img_size=(256, 256), balance_batches=False, **kwargs):
    """
    Generator to train a model on images.

    # Params
    - data_df : filenames
    - data_folder : folder where data resides. Should have structure
                    `data_folder/img_name`
    - labels : list of labels in onehot
    - batch_size : number of images per batch
    - shuffle : present images in random order (each epoch)
    - bboxes : A dictionary img_name -> (x, y, width, height). If this
               is given, the image will be cropped to this region.
    - augmentation : perform data augmentation
    - img_size : sample patches of this size from the image and mask
    - balance_batches : If true, balances batches so each class is 
                        equally represented
    - kwargs : passed to the preprocess function

    # Returns
    - batch of images (batch_size, 3, img_size[0], img_size[1])
    - batch of labels (batch_size, len(labels))
    """
    n_classes = labels.shape[1]
    
    while True:
        data = data_df
        n = len(data)
        if shuffle:
            combined = list(zip(data, labels))
            random.shuffle(combined)
            data[:], labels[:] = zip(*combined)
        data = list(data)

        # Double to allow for larger batch sizes
        data += data
        i = 0
        while i + batch_size - 1 < n:
            img_batch = np.zeros((batch_size, 3) + img_size, dtype=np.float32)
            label_batch = np.zeros((batch_size, n_classes), dtype=np.uint8)
            label_count = np.zeros(n_classes)
            j = 0 
            while j < batch_size:
                if i >= n:
                    i = 0
                img_name = data[i]
                label = labels[i]
                i += 1
                lab_nr = np.argmax(label)
                
                if balance_batches and label_count[lab_nr] >= batch_size / n_classes:
                    continue
                label_count[lab_nr] += 1
                img_name = img_name + ".jpg"
                img_path = os.path.join(data_folder, img_name)
                img = load_image(img_path)
                img = img[:,:,:3]
                img = preprocess(img, target_size=img_size, 
                                augmentation=augmentation, 
                                zero_center=False, scale=1./255.,
                                **kwargs)
                img_batch[j] = img
                label_batch[j] = label
                j += 1
            yield img_batch, label_batch
        

def get_test_data(files, batch_size=4, img_size=(720, 1280), **kwargs):
    """
    Generator for test data

    # Params
    - files : list of files (e.g., output of glob)
    - batch_size : number of images per batch
    - img_size : size to resize images to (height, width)
    - kwargs : passed to the preprocess function

    # Returns
    - batches of (batch_size, 3, img_size[0], img_size[1])
    """
    i = 0
    n = len(files)
    # cycle to avoid batches not lining up with dataset size
    files = files + files
    while True:
        batch = np.zeros((batch_size, 3) + img_size)
        for j in range(batch_size):
            img = load_image(files[i])
            img = preprocess(img, target_size=img_size, augmentation=False, 
                             zero_center=True, scale=1./255., **kwargs)
            batch[j] = img
            i = (i + 1) % n
        yield batch
