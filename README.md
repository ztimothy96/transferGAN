# TransferGAN

This repo holds a Pytorch adaptation of [Transferring GANs: generating images from limited data](https://arxiv.org/abs/1805.01677) (EECV 2018). Also supports training with EWC loss, as described in [Few-shot Image Generation with Elastic Weight Consolidation](https://arxiv.org/abs/2012.02780) (NeurIPS 2020). May add more experiments later.


## Training
Here are samples from 10000 iterations of fine-tuning from ImageNet to LSUN Bedrooms.

![10000_iters](https://github.com/ztimothy96/transferGAN/blob/main/training_10000_iters.gif)

## Dependencies
- Python 3

- Torch 1.7.1

- Numpy 1.19.4

- Torchvision 0.8.2

- Imageio 2.9.0 (for saving training gif)

- Matplotlib 3.3 (for plotting losses)

## Sources and inspiration
Much of this codebase is indebted to implementation ideas from Emilien Dupont's 
Pytorch [implementation](https://github.com/EmilienDupont/wgan-gp) of WGAN-GP 
and Naoto Inoue's Pytorch [implementation](https://github.com/naoto0804/pytorch-AdaIN) of AdaIN. 
