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


if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        print('Usage:\npython %s class_name [extension]')
    else:
        if sys.argv[2] != '':
            extract_class(sys.argv[1], sys.argv[2])
        else:
            extract_class(sys.argv[1])