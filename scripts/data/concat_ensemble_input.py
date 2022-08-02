# ------------------------------------------------------------------------
# Copyright (c) 2022 Woon-Ha Yeo <woonhahaha@gmail.com>.
# Copyright (c) 2022 Wang-Taek Oh <mm0741@naver.com>.
# ------------------------------------------------------------------------

import os
from os import path as osp
from glob import glob

from tqdm import tqdm
import cv2
import numpy as np


def create_dir(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


def main():
    basedir1 = './results/CAIR_S-width64-test-TTA-Ensemble'
    basedir2 = './results/CAIR_S-width32-test-TTA-Ensemble'
    basedir3 = './results/CAIR_M-width32-test-TTA-Ensemble'
    resultdirs = ['visualization/iffi-trainset', 'visualization/iffi-testset']
    concatdirs = ['./datasets/IFFI/train_ensemble/input', './datasets/IFFI/test_ensemble/input']
    
    create_dir(concatdirs[0])
    create_dir(concatdirs[1])

    for resultdir, concatdir in zip(resultdirs, concatdirs):
        file_list = glob(osp.join(basedir1, resultdir, '*.png'))
        # print(file_list)

        for f in tqdm(file_list):
            basename = osp.basename(f)
            
            image1 = cv2.imread(f)
            image2 = cv2.imread(osp.join(basedir2, resultdir, basename))
            image3 = cv2.imread(osp.join(basedir3, resultdir, basename))
            
            concat_image = np.concatenate((image1, image2, image3), axis=2)
            
            np.save(osp.join(concatdir, basename.split('.')[0] + '.npy'), concat_image)


if __name__ == '__main__':
    main()