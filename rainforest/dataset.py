import pandas as pd

labels = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']
train = pd.read_csv('../data/train.csv')
validation = pd.read_csv('../data/validation.csv')
