import numpy as np
import sys
import os
import skimage.io as sio
import joblib as jl
import matplotlib.pyplot as plt
sys.path.insert(0, '../')
import paths
from tqdm import tqdm
from skimage import img_as_ubyte
from spectral import imshow, get_rgb
from sklearn.preprocessing import MinMaxScaler

# https://www.kaggle.com/fppkaggle/making-tifs-look-normal-using-spectral-fork
# This link gives a proper scaler


def convert_tif_pkl(tif_folder='..' + paths.DATA_FOLDER + '/train-tif-sample',
                    jpg_folder='..' + paths.DATA_FOLDER + '/train-jpg-sample'):
    img = np.zeros((256, 256, 3))
    scaler = MinMaxScaler(feature_range=(0, 255))
    for f in tqdm(os.listdir(tif_folder)):
        tif_img = sio.imread(tif_folder + '/' + f)
        fil_no = os.path.splitext(f)[0]
        jpg_img = sio.imread(jpg_folder + '/' + fil_no + '.jpg')
        img2 = get_rgb(tif_img, [2, 1, 0])
        rescaleIMG = np.reshape(img2, (-1, 1))
        rescaleIMG = scaler.fit_transform(rescaleIMG) 
        img2 = (np.reshape(rescaleIMG, img2.shape)).astype(np.uint8)
        img = img2
    print img
    plt.imshow(img)
    plt.show(img)

if __name__ == "__main__":
    print 'hi'
    convert_tif_pkl()
