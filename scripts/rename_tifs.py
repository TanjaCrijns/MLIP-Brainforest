import pandas as pd
import os
import tqdm
import sys
sys.path.append('..')
import paths

data_folder = paths.DATA_FOLDER
os.mkdir(os.path.join(data_folder, 'test-tif-v2-fixed'))

mapping = pd.read_csv(os.path.join(data_folder, 'test_v2_file_mapping.csv'))
for old, new in tqdm.tqdm(mapping.itertuples(index=False), total=len(mapping)):
    os.rename(os.path.join(data_folder, 'test-tif-v2', old), os.path.join(data_folder, 'test-tif-v2-fixed', new))
