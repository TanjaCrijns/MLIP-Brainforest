import numpy as np
import sys
import glob
import os
import skimage.io as sio
import joblib as jl
sys.path.insert(0, '../')
import paths

#TODO normaliseren
def convert_tif_pkl(folder='..' + paths.DATA_FOLDER + '/train-tif'):
    for f in os.listdir(folder):
        img = sio.imread(folder + '/' + f)
        jl.dump(img, folder + '/' + os.path.splitext(f)[0] + '.pkl')
    
if __name__ == "__main__":
    convert_tif_pkl()
