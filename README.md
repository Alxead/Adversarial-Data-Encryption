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

First download CIFAR-10 and put it in an appropriate directory (e.g.  ``./data/cifar10``). Then train a standard (not robsut) ResNet-50 as base classifier through the following command:

```
python -m robustness.main --dataset cifar --data ./data/cifar10 --adv-train 0 --arch resnet50 --out-dir ./logs/checkpoints/dir/ --exp-name resnet50
```

After training, the base classifier is saved at  ``./logs/checkpoints/dir/resnet50/checkpoint.pt.best`` ,it will be used to encrypt the data.

### 2. Encrypt data

To encrypt the original CIFAR-10, run:

```
python encrypt.py --orig_data ./data/cifar10 --enc_data ./data --resume_path  
./logs/checkpoints/dir/resnet50/checkpoint.pt.best --enc_method basic
```

Use `--orig_data` to specify the directory where original CIFAR-10 is saved. Use `--enc_data` to specify the directory where encrypted CIFAR-10 will be saved.  Resume the base classifier from `--resume_path` and use option `--enc_method` to specify the encryption method. We provide four encrytion methods: `basic`, `mixup`, `horiz`, `mixandcat`.The other parameters of the encryption process are set to the values used in our paper by default. If you want to change them, you can check `encrypt.py` for more details.



### 3. Validate Encryption method





## Citation

