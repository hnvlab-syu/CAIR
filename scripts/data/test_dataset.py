import os
from glob import glob
import shutil
import argparse


def main(args):
    resultdir = args.resultdir
    basedir = os.path.join(resultdir, 'visualization/iffi-test')
    files = glob(os.path.join(basedir, '*'))

    savedir = os.path.join(resultdir, 'IFFI-dataset-lr-test')

    for file in sorted(files):
        basename = os.path.basename(file)
        savedir_num = os.path.join(savedir, basename.split('_')[0])
        if not os.path.exists(savedir_num):
            os.makedirs(savedir_num)
        shutil.copy(file, savedir_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resultdir', type=str, required=True, help='Path', default='./CAIR_S-width64')
    args = parser.parse_args()

    main(args)