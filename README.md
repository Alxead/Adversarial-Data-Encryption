# Adversarial-Data-Encryption

PyTorch implementation of paper "Adversarial Attack for Data Encryption".

Our implementation is based on these repositories:

- [robustness](https://github.com/MadryLab/robustness)

- [CIFAR-ZOO](https://github.com/BIGBALLON/CIFAR-ZOO)

### Abstract

In the big data era, many organizations face the dilemma of data sharing. Regular data sharing is often necessary  for human-centered discussion and communication, especially in medical scenarios. 
However, unprotected data sharing may also lead to  data leakage. Inspired by adversarial attack, 
we propose a method for data encryption, so that for human beings the encrypted data look identical to the original version,  but for machine learning methods they are misleading.

<img src="https://github.com/Alxead/Adversarial-Data-Encryption/blob/master/images/mainfig.png" width="600" alt="mainfig"/>





## Getting Started

### Requirements

`robustness` is a package [MadryLab](http://madry-lab.ml/) created to make training, evaluating, and exploring neural networks flexible and easy.  We mainly use `robustness` in the next first step (1. train a base classifier) and second step (2. encrypt data) .



### 1. Train a base classifier

First download the CIFAR-10 and put it in an appropriate directory (e.g.  ``./data/cifar10``). Then train a standard (not robsut) ResNet-50 through the following command:

```
python -m robustness.main --dataset cifar --data ./data/cifar10 --adv-train 0 --arch resnet50 --out-dir ./logs/checkpoints/dir/ --exp-name resnet50
```



### 2. Encrypt data



### 3. Validate Encryption method





## Citation

