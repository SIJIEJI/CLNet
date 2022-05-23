## Overview

This is the PyTorch implementation of the paper [CLNet: Complex Input Lightweight Neural
Network designed for Massive MIMO CSI Feedback](https://ieeexplore.ieee.org/document/9497358).
If you feel this repo helpful, please cite our paper:

```
@article{ji2021clnet,
  title={CLNet: Complex Input Lightweight Neural Network designed for Massive MIMO CSI Feedback},
  author={Ji, Sijie and Li, Mo},
  journal={IEEE Wireless Communications Letters},
  year={2021},
  publisher={IEEE}
  doi={10.1109/LWC.2021.3100493}}
}

```


## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.7
- [PyTorch >= 1.2 also compatible with PyTorch >= 1.7 with fft fix](https://pytorch.org/get-started/locally/)
- [thop](https://github.com/Lyken17/pytorch-OpCounter)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model. Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset in [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing), which is easier to use for the CSI feedback task; You can also download it from [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.

#### B. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── CLNet  # The cloned CLNet repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── main.py
├── COST2100  # The data folder
│   ├── DATA_Htestin.mat
│   ├── ...
├── Experiments
│   ├── checkpoints  # The checkpoints folder
│   │     ├── in_04.pth
│   │     ├── ...
│   ├── run.sh  # The bash script
...
```

## Train CLNet from Scratch

An example of run.sh is listed below. Simply use it with `sh run.sh`. It starts to train CLNet from scratch. Change scenario using `--scenario` and change compression ratio with `--cr`.

``` bash
python /home/CLNet/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --epochs 1000 \
  --batch-size 200 \
  --workers 8 \
  --cr 4 \
  --scheduler cosine \
  --gpu 0 \
  2>&1 | tee log.out
```

## Results and Reproduction

#### A. Model Complexity

The params and flops are directly caculated by [thop](https://github.com/Lyken17/pytorch-OpCounter). If you use this repo's code directly, the model complexity will be printed to the trainning log. A [sample training log](https://www.dropbox.com/sh/qhqknm60i97a966/AABip4HD4lw4_BdfuM7NtCGWa?dl=0) for your reference. The flops reported in the paper are caculated by fvcore to align with other SOTA works. The fvcore caculator didn't count the BN layer in, therefore it's less than thop. 

 | Compression Ratio | #Params | Flops | 
 | :--: | :--: | :--: | 
 | 1/4 | 2102K | 4.42M | 
 | 1/8 | 1053K | 3.37M |
 | 1/16 | 528.7K | 2.85M | 
 | 1/32 | 266.5K | 2.58M | 
 | 1/64 | 135.4K | 2.45M | 
 


#### B. Performance



The NMSE result reported in the paper as follow:

|Scenario | Compression Ratio | NMSE | Checkpoints
|:--: | :--: | :--: | :--: | 
|indoor | 1/4 | -29.16 |  in4.pth |
|indoor | 1/8 |  -15.60|  in8.pth|
|indoor | 1/16 | -11.15 |  in16.pth|
|indoor | 1/32 | -8.95 |  in32.pth|
|indoor | 1/64 | -6.34 |  in64.pth|
|outdoor | 1/4 | -12.88 | out4.pth|
|outdoor | 1/8 | -8.29 |  out8.pth|
|outdoor | 1/16 | -5.56 |  out16.pth|
|outdoor | 1/32 | -3.49 |  out32.pth|
|outdoor | 1/64 | -2.19 |  out64.pth|

If you want to reproduce our result, you can directly download the corresponding checkpoints from [Dropbox](https://www.dropbox.com/sh/qhqknm60i97a966/AABip4HD4lw4_BdfuM7NtCGWa?dl=0)


**To reproduce all these results, simple add `--evaluate` to `run.sh` and pick the corresponding pre-trained model with `--pretrained`.** An example is shown as follows.

``` bash
python /home/CLNet/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --pretrained './checkpoints/in4.pth' \
  --evaluate \
  --batch-size 200 \
  --workers 0 \
  --cr 4 \
  --cpu \
  2>&1 | tee test_log.out

```

## Acknowledgment

This repository is modified from the [CRNet open source code](https://github.com/Kylin9511/CRNet). Thanks Zhilin for his amazing work.
Thanks Chao-Kai Wen and Shi Jin group for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet) 

