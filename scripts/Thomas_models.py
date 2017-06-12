import sys
import os

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

sys.path.append('..')

from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPool2D
from keras.layers import Input
from keras.applications.vgg16 import preprocess_input
from keras.layers.core import Flatten, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from rainforest.data import get_class_data, get_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import fbeta_score
from spectral import get_rgb, ndvi
import paths
from skimage.transform import resize
import keras.backend as K
from skimage import io
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from shutil import copy2


def add_contraction_step(instance, depth):
    pass


def add_expansion_step(instance, depth):
    pass


def unetmodel(input_shape=(3, 256, 256), depth=3, branching_factor=3):
    input_layer_name = 'input'
    inputs = Input(shape=input_shape, name=input_layer_name)
    network = {input_layer_name: inputs}

    for i in xrange(depth):
        add_contraction_step(instance=network, depth=i)

    for i in xrange(depth):
        add_expansion_step(instance=network, depth=i)


def simple_model(input_shape=(1, 256, 256)):
    inp = Input(shape=input_shape)
    network = Conv2D(32, 3, kernel_initializer='glorot_uniform', activation='relu')(inp)
    network = Conv2D(64, 3, kernel_initializer='glorot_uniform', activation='relu')(network)
    network = MaxPool2D((2, 2))(network)
    network = Conv2D(64, 3, kernel_initializer='glorot_uniform', activation='relu')(network)
    network = Conv2D(64, 3, kernel_initializer='glorot_uniform', activation='relu')(network)
    network = MaxPool2D((2, 2))(network)
    network = Conv2D(64, 3, kernel_initializer='glorot_uniform', activation='relu')(network)
    network = MaxPool2D((2, 2))(network)
    network = Conv2D(128, 3, kernel_initializer='glorot_uniform', activation='relu')(network)
    network = MaxPool2D((2, 2))(network)
    network = Conv2D(256, 3, kernel_initializer='glorot_uniform', activation='relu')(network)
    network = Flatten()(network)
    network = Dense(1024, activation='relu')(network)
    network = Dropout(0.25)(network)
    network = Dense(512, activation='relu')(network)
    network = Dropout(0.25)(network)
    network = Dense(2, activation='softmax')(network)
    return inp, network


def test_generator(data_df, preprocess_func, batch_size=32, target_size=(5, 256, 256), subfolder='train-tif-v2'):
    index = 0
    while True:
        X_batch = np.zeros((batch_size, target_size[0], target_size[1], target_size[2]), dtype=np.float32)
        y_batch = np.zeros(batch_size, dtype=np.uint8)
        for k in range(batch_size):
            img_data = data_df.iloc[index]
            img = io.imread(os.path.join(paths.DATA_FOLDER, subfolder, img_data.image_name + '.tif')).transpose(2, 0, 1)
            img = preprocess_func(img, target_size=target_size)
            X_batch[k] = img
            y_batch[k] = img_data[1:].values
            index += 1
        yield X_batch


def data_generator(data_folder, preprocess_func, batch_size=32, target_size=(5, 256, 256), classes=[],
                   dist_mode='random',
                   dim_ordering='channels_first', ):
    chunk = batch_size / len(classes)
    rem = batch_size % len(classes)  # later fixen (je kan nu alleen batchsizes nemen die deelbaar zijn door je classes)
    classdict = {}

    for subdir in classes:
        subpath = os.path.join(data_folder, subdir)
        classdict[subdir] = [os.path.join(subpath, fname) for fname in os.listdir(subpath)]
    index = 0
    while True:
        X_batch = np.zeros((batch_size, target_size[0], target_size[1], target_size[2]), dtype=np.float32)
        y_batch = np.zeros(batch_size, dtype=np.uint8)
        j = 0
        for k in classdict:
            if dist_mode == 'random':
                images = np.random.choice(classdict[k], size=chunk, replace=False, p=None)
            elif dist_mode == 'ring':
                l = len(classdict[k])
                indices = range(index, index + chunk)
                indices = [x % l for x in indices]
                images = [classdict[k][i] for i in indices]
            for im in images:
                img = io.imread(im).transpose(2, 0, 1)
                img = preprocess_func(img, target_size=target_size)
                X_batch[j] = img
                y_batch[j] = classes.index(k)
                j += 1
        index += chunk
        y_batch = to_categorical(y_batch, num_classes=len(classes))
        yield X_batch, y_batch


def preprocess_input(X):
    return X


def preprocess_generator(gen):
    for X, y in gen:
        yield preprocess_input(X) / 255., y


def preprocess_image1(img, target_size):
    if target_size:
        img = resize(img, (img.shape[0], target_size[1], target_size[2]), order=1)
    img_rgb = get_rgb(img.transpose(1, 2, 0), [2, 1, 0])  # R-G-B
    rescaleimg = np.reshape(img_rgb, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 255))
    rescaleimg = scaler.fit_transform(rescaleimg)  # .astype(np.float32)
    img_scaled = (np.reshape(rescaleimg, img_rgb.shape)) / 255.0
    img_scaled = img_scaled.transpose(2, 0, 1)
    img_nir = get_rgb(img.transpose(1, 2, 0), [3, 2, 1])  # NIR-R-G
    img_nir_red = (img_nir[:, :, 0] - img_nir[:, :, 1]) / (img_nir[:, :, 0] + img_nir[:, :, 1] + np.finfo(float).eps)  # (NIR - RED) / (NIR + RED)
    img_nir_red = np.expand_dims(np.clip(img_nir_red, -1, 1), axis=0)
    img_nir_green = (img_nir[:, :, 2] - img_nir[:, :, 0]) / (img_nir[:, :, 2] + img_nir[:, :, 0] + np.finfo(float).eps)  # (GREEN - NIR) / (GREEN + NIR)
    img_nir_green = np.expand_dims(np.clip(img_nir_green, -1, 1), axis=0)

    return np.concatenate((img_scaled, img_nir_red, img_nir_green), axis=0)


def transfer_images(from_dir, to_dir, data, extension='tif'):
    if not os.path.exists(to_dir):
        os.mkdir(to_dir)
        for f in tqdm(data):
            copy2(os.path.join(paths.DATA_FOLDER, from_dir, f + '.' + extension), to_dir)


def fscore(y_true, y_pred):
    y_pred = y_pred > 0.5
    beta = 2
    smooth = 1e-6
    recall = (K.sum(y_true * y_pred, axis=1) + smooth) / (K.sum(y_true, axis=1) + smooth)
    precision = (K.sum(y_true * y_pred, axis=1) + smooth) / (K.sum(y_pred, axis=1) + smooth)
    return K.mean(((1 + beta ** 2) * (precision * recall) + smooth) / (beta ** 2 * precision + recall + smooth))


if __name__ == "__main__":
    input_shape = (5, 128, 128)
    inp, output = simple_model(input_shape=input_shape)
    beta = 2
    smoothing = 1e-6
    label = 'habitation'
    extension = 'tif'
    model_path = 'D:/MLIP-data/result/{}.hdf5'.format(label)
    csv_logger = CSVLogger('D:/MLIP-data/result/log.csv')
    lr_plateau = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5)
    checkpoint = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
    model = Model(inputs=inp, outputs=output)
    if os.path.exists(model_path):
        model.load_weights(model_path)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', fscore])
    data_folder = paths.DATA_FOLDER
    validation_folder = paths.DATA_FOLDER

    # val_dir = os.path.join(paths.DATA_FOLDER, 'val-tif')
    # if not os.path.isdir(val_dir):
    #     os.mkdir(val_dir)
    #     val_data = get_data(train=False)
    #     print val_data
    #     for f in tqdm(val_data.image_name, total=len(val_data)):
    #         copy2(os.path.join(paths.DATA_FOLDER, 'train-tif', f + '.' + extension), val_dir)

    train_data = get_class_data(train=True, label=label)
    val_data = get_class_data(train=False, label=label)
    train_pos = train_data[train_data[label] == 1]
    train_neg = train_data[train_data[label] == 0].sample(len(train_pos))
    val_pos = val_data[val_data[label] == 1]
    val_neg = val_data[val_data[label] == 0].sample(len(val_pos))

    tra_pos_folder = os.path.join(paths.DATA_FOLDER, 'tra_' + label)
    tra_neg_folder = os.path.join(paths.DATA_FOLDER, 'tra_negative_' + label)
    val_pos_folder = os.path.join(paths.DATA_FOLDER, 'val_' + label)
    val_neg_folder = os.path.join(paths.DATA_FOLDER, 'val_negative_' + label)

    image_folder = 'train-tif-v2'
    transfer_images(image_folder, tra_pos_folder, train_pos.image_name)
    transfer_images(image_folder, tra_neg_folder, train_neg.image_name)
    transfer_images (image_folder, val_pos_folder, val_pos.image_name)
    transfer_images(image_folder, val_neg_folder, val_neg.image_name)

    train_generator = data_generator(data_folder, preprocess_func=preprocess_image1, target_size=input_shape,
                                     classes=['tra_negative_' + label, 'tra_' + label], dist_mode='ring')

    val_generator = data_generator(validation_folder, preprocess_func=preprocess_image1, target_size=input_shape,
                                   classes=['val_negative_' + label, 'val_' + label], dist_mode='ring')
    # train_generator = preprocess_generator(train_generator)
    # val_generator = preprocess_generator(val_generator)

    model.fit_generator(train_generator, steps_per_epoch=50, epochs=40, callbacks=[csv_logger,lr_plateau,checkpoint],
                        validation_data=val_generator, validation_steps=20)
    val_data = get_class_data(train=False, label=label)
    test_generator = test_generator(val_data, preprocess_func=preprocess_image1, target_size=input_shape)
    batch_size = 32
    steps = np.floor(len(val_data) / 32)
    print steps
    print len(val_data)
    result = model.predict_generator(test_generator, steps-1, verbose=1)

    for threshold in np.arange(0.1, 0.6, 0.02):
        y_true = val_data.iloc[:len(result), 1:].values
        y_true = to_categorical(y_true, num_classes=2)
        y_pred = result > threshold
        print threshold, 'f2 score:', fbeta_score(y_true, y_pred, 2, average='samples')