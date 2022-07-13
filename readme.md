## AIM Workshop and Challenges @ ECCV 2022

### Challenges
- [Instagram Filter Removal Challenge](https://codalab.lisn.upsaclay.fr/competitions/5081#learn_the_details)
- [Compressed Input Super-Resolution Challenge - Track 1 Image](https://codalab.lisn.upsaclay.fr/competitions/5076#learn_the_details)

### Installation
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

### Training
#### Instagram Filter Removal
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/IFFI/IFFI.yml --launcher pytorch
```

#### Compressed Input Super-Resolution 
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4320 basicsr/train.py -opt options/train/CompressedInputSR/NAFNetSR-B_x4.yml --launcher pytorch
```


