'''
  | - datasets
  |  | - IFFI
  |  |  | - IFFI-dataset-train
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

 1. Get all train data from 'datasets/IFFI/IFFI-dataset-train'
     and copy IG filtered images to 'train/input' folder,
     and copy Original image to 'train/target' folder.
 2. Get all valid data
     and copy to 'val/input' folder.
'''

import os
from glob import glob
import shutil
import argparse

import cv2
from tqdm import tqdm

from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def create_lmdb_for_iffi(basedir):
    """Create lmdb files for IFFI train dataset."""

    folder_path = os.path.join(basedir, 'train/input')
    lmdb_path = os.path.join(basedir, 'train/input.lmdb')
    img_path_list, keys = prepare_keys_iffi(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = os.path.join(basedir, 'train/target')
    lmdb_path = os.path.join(basedir, 'train/target.lmdb')
    img_path_list, keys = prepare_keys_iffi(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def prepare_keys_iffi(folder_path):
    """Prepare image path list and keys for IFFI dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='jpg', recursive=False)))
    keys = [img_path.split('.jpg')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


def create_trainset(basedir, ensemble):
    folders = ['IFFI-dataset-train', 'IFFI-dataset-lr-train']
    for folder in folders:
        path_list = glob(os.path.join(basedir, folder, '*/*'), recursive=True)

        for file in tqdm(path_list):
            hr_lr = folder.split('-')[2]
            if hr_lr != 'lr':
                if not ensemble:
                    hr_lr = 'hr'
                    ensem_root = ''
                else:
                    ensem_root = '_ensemble'
            basename = os.path.basename(file)

            if 'Original' not in basename:
                if hr_lr != 'train':
                    original_file = os.path.join(os.path.dirname(file), basename.split('_')[0] + '_Original.jpg')
                    shutil.copy(original_file, os.path.join(basedir, 'train', f'target{ensem_root}', hr_lr + '_' + os.path.basename(file)))
                    shutil.copy(file, os.path.join(basedir, 'train', f'input{ensem_root}', hr_lr + '_' + os.path.basename(file)))
                if hr_lr in ['hr', 'train']:
                    img = cv2.imread(file)
                    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(basedir, 'train', f'target{ensem_root}', 'hlr' + '_' + os.path.basename(file)), resized)
                    cv2.imwrite(os.path.join(basedir, 'train', f'input{ensem_root}', 'hlr' + '_' + os.path.basename(file)), resized)


def main(args):
    basedir = args.basedir
    ensemble = args.ensemble
    
    # Train
    os.makedirs(os.path.join(basedir, 'train/input'), exist_ok=True)
    os.makedirs(os.path.join(basedir, 'train/target'), exist_ok=True)
    if ensemble:
        os.makedirs(os.path.join(basedir, 'train/input_ensemble'), exist_ok=True)
        os.makedirs(os.path.join(basedir, 'train/target_ensemble'), exist_ok=True)
        create_trainset(basedir, True)

    create_trainset(basedir, False)
    create_lmdb_for_iffi(basedir)
                
    # Validation
    # os.makedirs(os.path.join(basedir + 'val/input'), exist_ok=True)         
    # val_path_list = glob(os.path.join(basedir, 'IFFI-dataset-lr-valid', '*/*'), recursive=True)
    # for file in val_path_list:
    #     basename = os.path.basename(file)
    #     shutil.copy(file, os.path.join(basedir, 'val/input' + os.path.basename(file)))

    # Test
    os.makedirs(os.path.join(basedir, 'test/input'), exist_ok=True)
    test_path_list = glob(os.path.join(basedir, 'IFFI-dataset-lr-challenge-test-wo-gt', '*/*'), recursive=True)
    for file in test_path_list:
        shutil.copy(file, os.path.join(basedir, 'test/input/' + os.path.basename(file)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, required=True, help='Path', default='./datasets/IFFI')
    parser.add_argument('--ensemble', type=str2bool, required=True, help='Create Ensemble Dataset yes or no', default=False)
    args = parser.parse_args()

    main(args)
    