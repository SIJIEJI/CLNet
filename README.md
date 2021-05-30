## Overview

This is the PyTorch implementation of paper [CLNet: Complex Input Lightweight Neural
Network designed for Massive MIMO CSI Feedback](link).
If you feel this repo helpful, please cite our paper:

```

```


## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.7
- [PyTorch >= 1.2](https://pytorch.org/get-started/locally/)
- [thop](https://github.com/Lyken17/pytorch-OpCounter)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model. Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset in [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing), which is easier to use for the CSI feedback task; You can also download it from [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.

#### B. Checkpoints Downloading

The model checkpoints should be downloaded if you would like to reproduce our result. All the checkpoints files can be downloaded from [Dropbox]()

#### C. Project Tree Arrangement

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

An example of run.sh is listed below. Simply use it with `sh run.sh`. It will start advanced scheme aided CRNet training from scratch. Change scenario using `--scenario` and change compression ratio with `--cr`.

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

The params and flops are directly caculated by [thop](https://github.com/Lyken17/pytorch-OpCounter)

 | Compression Ratio | #Params | Flops | 
 | :--: | :--: | :--: | 
 | 1/4 | 2102K | 4.42M | 
 | 1/8 | 1053K | 3.37M |
 | 1/16 | 528.7K | 2.85M | 
 | 1/32 | 266.5K | 2.58M | 
 | 1/64 | 135.4K | 2.45M | 
 


#### B. Performance

The paper reports NMSE result can be reproduced by the follow:

|Scenario | Compression Ratio | NMSE | Checkpoints
|:--: | :--: | :--: | :--: | 
|indoor | 1/4 |  |  in_04.pth |
|indoor | 1/8 |  |  in_08.pth|
|indoor | 1/16 |  |  in_16.pth|
|indoor | 1/32 |  |  in_32.pth|
|indoor | 1/64 |  |  in_64.pth|
|outdoor | 1/4 |  | out_04.pth|
|outdoor | 1/8 |  |  out_08.pth|
|outdoor | 1/16 |  |  out_16.pth|
|outdoor | 1/32 |  |  out_32.pth|
|outdoor | 1/64 |  |  out_64.pth|

As aforementioned, we provide model checkpoints for all the results. Our code library supports easy inference. 

**To reproduce all these results, simple add `--evaluate` to `run.sh` and pick the corresponding pre-trained model with `--pretrained`.** An example is shown as follows.

``` bash
python /home/CRNet/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --pretrained './checkpoints/in_04' \
  --evaluate \
  --batch-size 200 \
  --workers 0 \
  --cr 4 \
  --cpu \
  2>&1 | tee log.out

```

## Acknowledgment

This repository is modified from the [CRNet open source code](https://github.com/Kylin9511/CRNet). Thanks Zhilin for his amazing work.
Thanks Chao-Kai Wen and Shi Jin group for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet) 

