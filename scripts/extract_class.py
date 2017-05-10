import sys
sys.path.append('..')

import paths
import pandas as pd
import os
import shutil
from skimage import io

def extract_class(class_name, extension='jpg'):
    src_dir = os.path.join(paths.DATA_FOLDER, 'train-jpg')
    dest_dir = os.path.join(paths.DATA_FOLDER, class_name)
    os.mkdir(dest_dir)
    label_csv = pd.read_csv(os.path.join(paths.DATA_FOLDER, 'train_v2.csv'))
    labels = label_csv.tags.str.split(' ').values
    for image_name, image_labels in zip(label_csv.image_name.values, labels):
        if class_name in image_labels:
            img = io.imread(os.path.join(src_dir, image_name) + '.jpg')
            io.imsave(os.path.join(dest_dir, image_name) + '.' + extension, img)

def extract_negative(class_name, n_images, extension='jpg'):
    src_dir = os.path.join(paths.DATA_FOLDER, 'train-jpg')
    dest_dir = os.path.join(paths.DATA_FOLDER, 'negative_' + class_name)
    os.mkdir(dest_dir)
    label_csv = pd.read_csv(os.path.join(paths.DATA_FOLDER, 'train_v2.csv'))
    labels = label_csv.tags.str.split(' ').values
    i = 0
    for image_name, image_labels in zip(label_csv.image_name.values, labels):
        if class_name not in image_labels:
            i += 1
            img = io.imread(os.path.join(src_dir, image_name) + '.jpg')
            io.imsave(os.path.join(dest_dir, image_name) + '.' + extension, img)
            if i >= n_images:
                return

if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        print('Usage:\npython %s class_name [extension]')
    else:
        if sys.argv[2] != '':
            extract_negative(sys.argv[1], 200, sys.argv[2])
        else:
            extract_class(sys.argv[1])