## CAIR: Multi-Scale Color Attention Network for Instagram Filter Removal

### Challenge
- AIM 2022 [Instagram Filter Removal Challenge](https://codalab.lisn.upsaclay.fr/competitions/5081#learn_the_details)

### Installation & Setting
This implementation is based on [NAFNet](https://github.com/megvii-model/NAFNet). 

```python
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
You can train on Instagram Filter Removal Challenge dataset by following these steps:
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

Your Instagram Filter Removal Challenge dataset to lmdb format. If you want to use ensemble learning, set '--ensemble true'.
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
# CAIR_S-width64
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4320 basicsr/test.py -opt ./options/test/Ensemble/CAIR_S-width64.yml --launcher pytorch
# CAIR_S-width32
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4319 basicsr/test.py -opt ./options/test/Ensemble/CAIR_S-width32.yml --launcher pytorch
python concat_ensemble_input.py
```
- Train ensemble network
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/Ensemble/EnsembleNet-nb3.yml --launcher pytorch
```
- Test ensemble network
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/Ensemble/EnsembleNet-nb3.yml --launcher pytorch
```