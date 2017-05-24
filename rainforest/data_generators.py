from __future__ import division

import os

import numpy as np
from preprocess import to_categorical as onehot

from preprocess import *
import data

def get_data(data_df, data_folder, labels, batch_size=32, shuffle=True, bboxes=None,
             augmentation=True, img_size=(256, 256), balance_batches=False, **kwargs):
    """
    Generator to train a model on images.

    # Params
    - data_df : DataFrame of filename and label for each image
    - data_folder : folder where data resides. Should have structure
                    `data_folder/label/img_name`
    - labels : list of labels
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
    n_classes = len(labels)
    
    while True:
        data = zip(data_df.filename.values, data_df.label.values)
        n = len(data)
        if shuffle:
            data = np.random.permutation(data)
        data = list(data)

        # Double to allow for larger batch sizes
        data += data
        i = 0
        labelcount = np.zeros(len(labels))
        while i < n:
            img_batch = np.zeros((batch_size, 3) + img_size, dtype=np.float32)
            label_batch = np.zeros((batch_size, n_classes), dtype=np.uint8)
            j = 0
            label_count = np.zeros(len(labels))
            while j < batch_size:
                img_name, label = data[i]
                i += 1
                lab_nr = labels.index(label)
                
                if balance_batches and label_count[lab_nr] >= batch_size / len(labels):
                    continue
                label_count[lab_nr] += 1
                img_path = os.path.join(data_folder, label, img_name)
                img = load_image(img_path)
                if bboxes is not None:
                    if img_name in bboxes:
                        boxes = bboxes[img_name]
                        if isinstance(boxes, list):
                            # if we have a list of boxes, choose a box at random
                            box_idx = np.random.randint(len(boxes))
                            x, y, width, height = boxes[box_idx]
                        else:
                            # otherwise it already is just 1 box
                            x, y, width, height = boxes
                    else :
                        # no bounding box found, choose random box
                        x = np.random.randint(img.shape[1] - 256)
                        y = np.random.randint(img.shape[0] - 256)
                        height, width = img_size

                    # Crop the image to the bounding box
                    x, y, width, height = [int(n) for n in [x, y, width, height]]
                    if x + width > img.shape[1]:
                        x = img.shape[1] - width
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    if y + height > img.shape[0]:
                        y = img.shape[0] - height
                    img = img[y:y+height, x:x+width]
                img = preprocess(img, target_size=img_size, 
                                augmentation=augmentation, 
                                zero_center=True, scale=1./255.,
                                **kwargs)
                img_batch[j] = img
                label_batch[j] = onehot(labels.index(label), len(labels))
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
            

def get_data_with_masks(files_and_masks, batch_size=1,
                        shuffle=True, augmentation=True, img_size=(256, 256),
                        **kwargs):
    """
    Generator to train a model on images with both segmentation and
    whole image label.

    # Params
    - data_df : DataFrame of filename and label for each image
    - bboxes : dictionary of 
               img_name -> [(x, y, width, height, class), ...]
    - data_folder : folder where data resides. Should have structure 
                    `data_folder/label/img_name`
    - batch_size : number of images per batch
    - shuffle : present images in random order (each epoch)
    - augmentation : perform data augmentation
    - img_size : sample patches of this size from the image and mask
    - kwargs : passed to the preprocess function

    # Returns
    - batch of images (batch_size, 3, img_size[0], img_size[1])
    - masks (batch_size, 2, img_size[0], img_size[1])
    """
    
    while True:
        data = files_and_masks
        n = len(data)
        if shuffle:
            data = np.random.permutation(data)
        data = list(data)

        # Double to allow for larger batch sizes
        data += data
        i = 0
        while i < n:
            img_batch = np.zeros((batch_size, 3) + img_size, dtype=np.float32)
            mask_batch = np.zeros((batch_size, 2) + img_size, dtype=np.uint8)
            for j in range(batch_size):
                img_path, annot_path = data[i]
                img = load_image(img_path)
                annotation = load_image(annot_path)
                annotation = annotation.sum(axis=2) > 0
                mask = np.zeros(img.shape[:2] + (2,), dtype=np.uint8)
                mask[:, :, 0] = ~annotation
                mask[:, :, 1] = annotation

                img, mask = preprocess(img, target_size=img_size, 
                                       augmentation=augmentation, mask=mask, 
                                       zero_center=True, scale=1./255.,
                                       **kwargs)

                img_batch[j] = img
                mask_batch[j] = mask
                i += 1
            yield img_batch, mask_batch

def get_data_with_bbox_coords(data_df, data_folder, bboxes, labels, batch_size=32, shuffle=True, 
                              augmentation=True, img_size=(256, 256), **kwargs):
    """
    Generator to train a model on images.
    Images which don't have a bounding box are simply skipped.

    # Params
    - data_df : DataFrame of filename and label for each image
    - data_folder : folder where data resides. Should have structure
                    `data_folder/label/img_name`
    - batch_size : number of images per batch
    - shuffle : present images in random order (each epoch)
    - augmentation : perform data augmentation
    - img_size : sample patches of this size from the image and mask
    - kwargs : passed to the preprocess function

    # Returns
    - batch of images (batch_size, 3, img_size[0], img_size[1])
    - batch of labels (batch_size, len(labels))
    """
    n_coords = 4
    n_classes = len(labels)
    
    while True:
        data = zip(data_df.filename.values, data_df.label.values)
        n = len(data)
        if shuffle:
            data = np.random.permutation(data)
        data = list(data)

        # Double to allow for larger batch sizes
        data += data
        i = 0
        while i < n:
            img_batch = np.zeros((batch_size, 3) + img_size, dtype=np.float32)
            bbox_batch = np.zeros((batch_size, n_coords), dtype=np.int32)
            label_batch = np.zeros((batch_size, n_classes), dtype=np.uint8)
            for j in range(batch_size):
                img_name, label = data[i]
                img_path = os.path.join(data_folder, label, img_name)
                img = load_image(img_path)
                if img_name not in bboxes:
                    bbox = (0, 0) + img_size
                else:
                    bbox = bboxes[img_name]
                if len(bbox) == 5:
                    bbox = bbox[:4]
                x, y, width, height = bbox
                # Make a mask from the bounding box, so we can apply
                # data augmentation to it. Later we will convert it back
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask[y:y+height, x:x+width] = 1

                img, mask = preprocess(img, target_size=img_size,
                                       augmentation=augmentation,
                                       zero_center=True, scale=1./255.,
                                       mask=mask, **kwargs)
                
                img_batch[j] = img
                bbox_batch[j] = bbox_from_segmentation(mask)
                label_batch[j] = onehot(labels.index(label), len(labels))
                i += 1
            yield img_batch, [label_batch, bbox_batch]
