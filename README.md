# BHViT
This code is an implementation of our work "BHViT: Binarized Hybrid Vision Transformer."

----------------------------------------------------------------------------------------------------------------------------------------------------------
# [BHViT: Binarized Hybrid Vision Transformer](https://arxiv.org/abs/2503.02394)

### Environment and Dependencies
Our code was tested with Python 3.11.7, Pytorch 2.5.1,and cuda 12.1  

Required python packages：
* PyTorch (version 2.5.1)
* numpy
* timm
* transformers 4.39.2
### Tips
   Any problem, please contact the first author (Email: gaotian970228@njust.edu.cn).

   Our pre-trained model can be downloaded at the following link
   * BHViT-small
     [somefp 70.1](https://drive.google.com/drive/folders/1K8W9LjFQIemG6Cc6xMXzmAOTgBuN9_8h)
     [binary68.4](https://drive.google.com/drive/folders/1K8W9LjFQIemG6Cc6xMXzmAOTgBuN9_8h)
   * BHViT-tiny
     [somefp 66.0](https://drive.google.com/drive/folders/1tuEdd8xkLSuwoordYdEl4VpKLRmP3xJO)
     [binary64.0](https://drive.google.com/drive/folders/1tuEdd8xkLSuwoordYdEl4VpKLRmP3xJO)
### Citation
If you find this work useful, please consider citing:
    @inproceedings{gao2025bhvit,
      title={BHViT: Binarized Hybrid Vision Transformer}, 
      author={Tian Gao and Zhiyuan Zhang and Yu Zhang and Huajun Liu and Kaijie Yin and Chengzhong Xu and Hui Kong},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2025},
          }
### License
Our code is released under the MIT License (see LICENSE file for details).
### Acknowledgement
Our code refers to binaryViT(https://github.com/Phuoc-Hoan-Le/BinaryViT) and DeiT(https://github.com/facebookresearch/deit).
### Teacher weights
1. [GSB-Vision-Transformer](https://github.com/IMRL/GSB-Vision-Transformer)包含了3个teacher权重，分别是：
    1. cifar100 teacher weights
    1. flower teacher weights
    1. chaoyang teacher weights

