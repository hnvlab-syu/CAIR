# CAIR: Multi-Scale Color Attention Network for Instagram Filter Removal (ECCV 2022 Workshop)
[Woon-Ha Yeo](https://scholar.google.com/citations?user=rzLqXnkAAAAJ&hl=ko&oi=sra), [Wang-Taek Oh](https://scholar.google.com/citations?user=lbm9wBQAAAAJ&hl=ko&oi=sra), Kyung-Su Kang, [Young-Il Kim](https://scholar.google.co.kr/citations?hl=ko&user=VJWlpfsAAAAJ), Han-Cheol Ryu

[![arXiv](https://img.shields.io/badge/arXiv-2208.14039-b31b1b.svg)](https://arxiv.org/abs/2208.14039)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://www.youtube.com/watch?v=4LIOKXfiQSE)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://www.slideshare.net/WoonHaYeo/cair-fast-and-lightweight-multiscale-color-attention-network-for-instagram-filter-removal)

### Challenge
- AIM 2022 [Instagram Filter Removal Challenge](https://codalab.lisn.upsaclay.fr/competitions/5081#learn_the_details)

![cair result](https://user-images.githubusercontent.com/49676680/208600943-613a7dfd-3e9f-4e94-8bce-6ff919657f94.png)

### Installation & Setting
This implementation is based on [NAFNet](https://github.com/megvii-research/NAFNet). 

```
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

```
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

## CAIR
### Data Prepration
You can download IFFI(Instagram Filter Fashion Image) dataset on [challenge website](https://codalab.lisn.upsaclay.fr/competitions/5081#learn_the_details)

You can train on IFFI dataset by following these steps:
<details><summary>Datasets directory structure</summary>

```
  datasets
  └──IFFI
     └──IFFI-dataset-train
     |  └──0
     |  └──1
     |  └──2
     |  └──...
     └──IFFI-dataset-lr-train
     |  └──0
     |  └──1
     |  └──2
     |  └──...
     └──IFFI-dataset-lr-challenge-test-wo-gt
        └──0
        └──1
        └──2
        └──...
```
</details>

Your Instagram Filter Removal Challenge dataset to lmdb format. If you want to use ensemble learning, set `--ensemble true`.
```
python scripts/data/dataset_to_lmdb.py --basedir ./datasets/IFFI --ensemble false
python scripts/data/dataset_to_lmdb.py --basedir ./datasets/IFFI --ensemble true
```

Result data into submission format
```
python scripts/data/test_dataset.py --resultdir ./results/model_name
```

### Train/Test
If you want to train/test other model, replace option file with others in [`options/`](https://github.com/HnV-Lab/CAIR/tree/main/options) folder.
- Train
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/CAIR/CAIR_M-width32.yml --launcher pytorch
```
- Test
```
# General test
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/CAIR/CAIR_M-width32.yml --launcher pytorch
#  TTA
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/CAIR-TTA/CAIR_M-width32.yml --launcher pytorch
```


## Ensemble learning
- Data preparation

  For ensemble learning, train and test set should be inferenced with three models, and then three images from three models should be contatenated.
```
# CAIR_M-width32
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt ./options/test/Ensemble/CAIR_M-width32.yml --launcher pytorch
# CAIR_S-width32
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4319 basicsr/test.py -opt ./options/test/Ensemble/CAIR_S-width32.yml --launcher pytorch
python scripts/data/concat_ensemble_input.py
```
- Train ensemble network
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/Ensemble/CAIR_Ensemble.yml --launcher pytorch
```
- Test ensemble network
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/Ensemble/CAIR_Ensemble.yml --launcher pytorch
```

## Pretrained Models
| name |PSNR|SSIM| pretrained models |
|:----|:----|:----|:----|
|CAIR-S|33.87|0.970|[Synology drive](http://gofile.me/6Z850/dZzINMTPs) |
|CAIR-M|34.39|0.971|[Synology drive](http://gofile.me/6Z850/vEO5jLxZs)  |  
|CAIR-Ensemble(CAIR*)|34.42|0.972|[Synology drive](http://gofile.me/6Z850/RrYI3fHgb) |

## Citation
```
@article{yeo2022cair,
  title={CAIR: Fast and Lightweight Multi-Scale Color Attention Network for Instagram Filter Removal},
  author={Yeo, Woon-Ha and Oh, Wang-Taek and Kang, Kyung-Su and Kim, Young-Il and Ryu, Han-Cheol},
  journal={arXiv preprint arXiv:2208.14039},
  year={2022}
}
```

## Contacts
If you have any question, please contact mm074111@gmail.com

## Acknowledgements
<details><summary>Expand</summary>

- https://github.com/XPixelGroup/BasicSR.git
- https://github.com/megvii-research/NAFNet.git
- https://github.com/swz30/CycleISP.git
