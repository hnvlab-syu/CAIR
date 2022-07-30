## AIM Workshop and Challenges @ ECCV 2022

### Challenges
- [Instagram Filter Removal Challenge](https://codalab.lisn.upsaclay.fr/competitions/5081#learn_the_details)

### Installation & Setting
This implementation is from [NAFNet](https://github.com/megvii-model/NAFNet). 

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

```
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Data Prepration
You can also train on Instagram Filter Removal Challenge dataset by following these steps:
```
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
  |  |  | - IFFI-dataset-lr-challenge-test-wo-gt
  |  |  |  | - 0
  |  |  |  | - 1
  |  |  |  | - 2
  |  |  |  | - ...
```
Your Instagram Filter Removal Challenge dataset to lmdb format. If you want to use ensemble learning, set '--ensemble true'.
```
python scripts/data/dataset_to_lmdb.py --basedir ./datasets/IFFI --ensemble false
python scripts/data/dataset_to_lmdb.py --basedir ./datasets/IFFI --ensemble true
```

Inferenced test data into 숫자 folders
```
python scripts/data/test_dataset.py --resultdir ./results/model_name
```

### Train/Test
#### Instagram Filter Removal
- Train
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/CAIR/CAIR_M-width32.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/Ensemble/EnsembleNet-nb3.yml --launcher pytorch

```
- Test
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/CAIR/CAIR_M-width32.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/CAIR-TTA/CAIR_M-width32.yml --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/Ensemble/EnsembleNet-nb3.yml --launcher pytorch
```

