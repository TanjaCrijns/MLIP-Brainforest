import pandas as pd
import sys
sys.path.insert(0, '..')
import paths

data_path = (paths.DATA_FOLDER)

labels = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
          'blow_down', 'clear', 'cloudy', 'conventional_mine', 'cultivation',
          'habitation', 'haze', 'partly_cloudy', 'primary', 'road',
          'selective_logging', 'slash_burn', 'water']

def get_class_data(train=True, label='weather'):
    '''
    Returns either the train or validation data with the class you want to 
    train. If the weather class all weather classes will be returned.
    The data is returned as a pandas dataframe.
    '''
    df = None
    if train:
        df = pd.read_csv(data_path + '/train.csv')
    else: # validation if not train
        df = pd.read_csv(data_path + '/validation.csv')

    if label == 'weather':
        df = df[['image_name', 'clear', 'cloudy', 'haze', 'partly_cloudy']]
    elif label in labels:
        df = df[['image_name', label]]
    else:
        raise ValueError('label should be a class label or weather') 
    
    return df

def get_data(train=True):
    '''
    Returns either the train or validation data (in pandas dataframe) with all
    the classes binarized
    '''
    df = None
    if train:
        df = pd.read_csv(data_path + '/train.csv')
    else: # validation if not train
        df = pd.read_csv(data_path + '/validation.csv')
    return df
