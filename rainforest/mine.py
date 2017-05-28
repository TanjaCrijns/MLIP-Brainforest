import numpy as np
import matplotlib as plt
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../rainforest')
from paths import DATA_FOLDER
from data import get_class_data
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Model
from scipy import misc
from sklearn.preprocessing import label_binarize
import keras.layers.core as C
from keras.models import Sequential


train = get_class_data(train=True, label='mines')
validation = get_class_data(train=False, label='mines')


# ### Info
# 
# ###### Validation:
#     artisial mine: 95, conventional mine: 26, total: 12144
# ###### Train:
#     artisial mine: 244, conventional mine: 74, total: 28335
#     
#     2 images in train set with both labels
#     1 image in val set with both labels

# data generator

def data_generator_balanced(data, label, length, train):
    '''
    Input:
        data: pandas object containing the data 
        label: label of data you want to generate images from
        length: batch length
        train: if train folder or test
    '''
    data = data[['image_name', label]]
    cnt = 0
    folder = '/train-jpg/'
    if not train:
        folder = '/test-jpg/'
    while True:
        batch = np.zeros((length, 3, 256, 256))
        labels = np.ones((50, 2))
        for i in range(length/2):  # get mine images
            img_name = np.random.choice(data[data[label] == 1]['image_name'].as_matrix())
            img = '..' + DATA_FOLDER + folder + img_name + '.jpg'
            batch[i, :, :, :] = misc.imread(img)[:,:,:3].transpose(2,0,1)
            # labels[i] = 1
            labels[i] = [0, 1]
        for i in range(length/2, length):  # get non mine images
            img_name = np.random.choice(data[data[label] == 0]['image_name'].as_matrix())
            img = '..' + DATA_FOLDER + folder + img_name + '.jpg'
            batch[i, :, :, :] = misc.imread(img)[:,:,:3].transpose(2,0,1)
            # labels[i] = 0
            labels[i] = [1, 0]
        
        # Shuffle batch
        idx = np.arange(length)
        np.random.shuffle(idx)
        batch = batch[idx]
        
        labels = np.array(labels[idx])
        yield batch, labels 


np.random.choice(train[train['conventional_mine'] == 1]['image_name'].as_matrix())


gen = data_generator_balanced(train, 'conventional_mine', 30, True)
val = data_generator_balanced(validation, 'conventional_mine', 30, True)


# augmentation generator
def augmentation_generator(data_gen):
    '''
    Input:
        data_gen: a data generator that generates batches of data that need to be augmented
    '''
    while True:
        yield None


def model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(3, 256, 256)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    return model

a = model()
a.compile(optimizer='adam', loss='categorical_crossentropy')


d = gen.next()
print d[0].shape
print d[1].shape


a.summary()

a.fit_generator(gen, 10, epochs=1, verbose=2, callbacks=None, validation_data=val, validation_steps=100)
