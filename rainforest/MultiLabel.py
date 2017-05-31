
# coding: utf-8

# In[56]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

import sys
sys.path.append('..')

import glob
import os

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn.metrics import fbeta_score

from keras.models import Sequential
from keras.layers import *
from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
import paths

from rainforest.data import get_data, labels
from rainforest.preprocess import preprocess
from rainforest.models.resnet import ResNet50
from rainforest.models.densenet import create_dense_net


# In[57]:

train_data = get_data(train=True)
val_data = get_data(train=False)


# In[58]:

batch_size=32
input_size=(64, 64)


# In[59]:

def data_generator(data_df, batch_size=32, target_size=(256, 256), shuffle=True, augmentation=True, subfolder='train-jpg'):
    n = len(data_df)
    while True:
        # Maybe shuffle
        data = data_df.sample(frac=1) if shuffle else data_df
        data = data.append(data, ignore_index=True)
        i = 0
        while i < n:
            X_batch = np.zeros((batch_size, target_size[0], target_size[1], 3) , dtype=np.float32)
            y_batch = np.zeros((batch_size, 17), dtype=np.uint8)
            
            for j in range(batch_size):
                img = data.iloc[i]
                img_path = os.path.join('..' + paths.DATA_FOLDER, subfolder, img.image_name+'.jpg')
                image = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
                image = preprocess(image, target_size=target_size, augmentation=augmentation,
                           hflip=True, vflip=True, scale=1./255., shift_x=3, shift_y=3, rot_range=5)
                image = np.transpose(image, (1, 2, 0))
                X_batch[j] = image
                y_batch[j] = img[1:].values
                i += 1
            
            yield X_batch, y_batch


# In[60]:

d = data_generator(train_data)
d.next()[0]


# In[61]:

def fb_score(beta=1, smooth=1e-6):
    
    def fscore(y_true, y_pred):
        y_pred = y_pred > 0.5
        recall = (K.sum(y_true * y_pred, axis=1) + smooth) / (K.sum(y_true, axis=1) + smooth)
        precision = (K.sum(y_true * y_pred, axis=1) + smooth) / (K.sum(y_pred, axis=1) + smooth)
        return K.mean( ((1+beta**2) * (precision*recall)+smooth) / (beta**2*precision+recall+smooth) )
    
    fscore.__name__ = 'F%d_score' % beta
    
    return fscore


# In[79]:




# In[62]:

def resnet_like():
    model = ResNet50(input_shape=(64, 64, 3), classes=17, classification='sigmoid', layer1_filters=32)
    return model


# In[63]:

def vgg_like():
    model = Sequential([
        Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', input_shape=(3,)+input_size),
        BatchNormalization(axis=1),
        Conv2D(32, 3, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(axis=1),
        MaxPool2D(),

        Conv2D(64, 3, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(axis=1),
        Conv2D(64, 3, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(axis=1),
        MaxPool2D(),

        Conv2D(128, 3, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(axis=1),
        Conv2D(128, 3, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(axis=1),
        MaxPool2D(),

        Flatten(),
        Dense(1024, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dense(17, activation='sigmoid')
    ])
    
    return model


# In[80]:

model = create_dense_net(17, (64, 64, 3), weight_decay=0)

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[81]:

train_gen = data_generator(train_data, batch_size=batch_size, target_size=input_size, shuffle=True, augmentation=True)
val_gen = data_generator(val_data, batch_size=batch_size, target_size=input_size, shuffle=False, augmentation=False)


# In[83]:

csv_logger = CSVLogger('log.csv')
lr_plateau = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.25)
checkpoint = ModelCheckpoint(filepath='models/brainforest/resnet16_64x64.hdf5', verbose=1, save_best_only=True)


# In[84]:

train_steps = len(train_data) // batch_size
val_steps = len(val_data) // batch_size
model.fit_generator(train_gen, train_steps, epochs=50, callbacks=[csv_logger, lr_plateau, checkpoint],
                    validation_data=val_gen, validation_steps=val_steps)


# In[15]:

model.load_weights('E:/Models/brainforest/resnet16_64x64.hdf5')

def strip_labels(gen):
    while True:
        imgs, _ = next(gen)
        yield imgs

val_steps = int(np.ceil(len(val_data) // batch_size)) + 1
val_gen = strip_labels(data_generator(val_data, batch_size=batch_size, target_size=input_size, shuffle=False))
preds = model.predict_generator(val_gen, val_steps)
preds = preds[:len(val_data)]


# In[16]:

for threshold in np.arange(0.1, 0.3, 0.02):
    y_true = val_data.iloc[:, 1:].values
    y_pred =  preds > threshold
    print threshold, 'f2 score:', fbeta_score(y_true, y_pred, 2, average='samples')


# In[13]:

test_files = glob.glob(os.path.join(paths.DATA_FOLDER, 'test-jpg', '*.jpg'))
test_files = [os.path.basename(os.path.splitext(f)[0]) for f in test_files]
test_data = pd.DataFrame(test_files, columns=['image_name'])
test_data['bogus_label'] = np.zeros(len(test_files))


# In[14]:

test_data.head()


# In[15]:

test_steps = int(np.ceil(len(test_data) // batch_size)) + 1
test_gen = strip_labels(data_generator(test_data, batch_size=batch_size, target_size=input_size, shuffle=False, subfolder='test-jpg'))
preds = model.predict_generator(test_gen, test_steps)
preds = preds[:len(test_data)]


# In[16]:

tpreds = preds > 0.16


# In[17]:

with open('submission2.csv', 'w') as file:
    file.write('image_name,tags\n')
    for img, pred in zip(test_files, tpreds):
        indices = np.flatnonzero(pred)
        tags = ' '.join([labels[i] for i in indices])
        file.write('%s,%s\n' % (img, tags))


# In[18]:

from sklearn.externals import joblib
joblib.dump(preds, 'multilabel2405_128x128.pkl')


# In[19]:

preds = (joblib.load('multilabel2405_128x128.pkl') + joblib.load('multilabel2405.pkl')) / 2


# In[20]:

tpreds = preds > 0.16


# In[21]:

with open('submission3.csv', 'w') as file:
    file.write('image_name,tags\n')
    for img, pred in zip(test_files, tpreds):
        indices = np.flatnonzero(pred)
        tags = ' '.join([labels[i] for i in indices])
        file.write('%s,%s\n' % (img, tags))


# In[ ]:



