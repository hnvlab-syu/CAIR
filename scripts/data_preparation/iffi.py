'''
Train 데이터를 모두 불러와서 필터 이미지는 input 폴더에 저장하고, Original 이미지는 target 폴더에 저장한다.
'''
import os
from glob import glob
import shutil

basedir = '/workspace/why/codalab/NTIRE22/NAFNet/datasets/IFFI/'

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