import os
import shutil

import kaggle

DATA_DIR = './data/'

if __name__ == '__main__':
    if not os.path.exists(os.path.join(DATA_DIR, 'val.X')):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('ambityga/imagenet100', path=DATA_DIR, unzip=True)
    if not os.path.exists(os.path.join(DATA_DIR, 'train.comb')):
        target = os.path.join(DATA_DIR, 'train.comb')
        os.mkdir(target)
        for i in range(1, 5):
            parent = os.path.join(DATA_DIR, f'train.X{i}')
            for d in os.listdir(parent):
                shutil.move(os.path.join(parent, d), target)
