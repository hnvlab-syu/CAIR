'''
  | - datasets
  |  | - IFFI
  |  |  | - IFFI-dataset-hr-train
  |  |  |  | - 0
  |  |  |  | - 1
  |  |  |  | - 2
  |  |  |  | - ...
  |  |  | - IFFI-dataset-lr-train
  |  |  |  | - 0
  |  |  |  | - 1
  |  |  |  | - 2
  |  |  |  | - ...
  |  |  | - IFFI-dataset-lr-valid
  |  |  |  | - 0
  |  |  |  | - 1
  |  |  |  | - 2
  |  |  |  | - ...

 1. Get all train data from 'datasets/IFFI/IFFI-dataset-hr-train'
     and copy IG filtered images to 'train/input' folder,
     and copy Original image to 'train/target' folder.
 2. Get all valid data
     and copy to 'val/input' folder.
'''

import os
from glob import glob
import shutil

basedir = './datasets/IFFI/'

# Train
folders = ['IFFI-dataset-hr-train', 'IFFI-dataset-lr-train']

for folder in folders:
    path_list = glob(os.path.join(basedir, folder, '*/*'), recursive=True)

    for file in path_list:
        hr_lr = folder.split('-')[2]
        basename = os.path.basename(file)
        if 'Original' not in basename:
            original_file = os.path.join(os.path.dirname(file), basename.split('_')[0] + '_Original.jpg')
            shutil.copy(original_file, os.path.join(basedir, 'train', 'target', hr_lr + '_' + os.path.basename(file)))
            shutil.copy(file, os.path.join(basedir, 'train', 'input', hr_lr + '_' + os.path.basename(file)))
            
# Validation            
val_path_list = glob(os.path.join(basedir, 'IFFI-dataset-lr-valid', '*/*'), recursive=True)
for file in val_path_list:
    shutil.copy(file, os.path.join(basedir, 'val/input'))