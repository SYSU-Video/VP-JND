<h1 align="center">VP-JND:Visual Perception Assisted Deep Picture-Wise Just Noticeable Difference Prediction Model for Image Compression </h1>

![GitHub stars](https://img.shields.io/github/stars/SYSU-Video/VP-JND?style=social)
[![Paper](https://img.shields.io/badge/Paper-TCSVT'25-b31b1b.svg)](https://ieeexplore.ieee.org/document/11112672)
![License](https://img.shields.io/github/license/SYSU-Video/VP-JND)
![Last commit](https://img.shields.io/github/last-commit/SYSU-Video/VP-JND)

Official code of "VP-JND:Visual Perception Assisted Deep Picture-Wise Just Noticeable Difference Prediction Model for Image Compression" \
[[paper]](https://ieeexplore.ieee.org/document/11112672) [[code]](https://github.com/SYSU-Video/VP-JND) \
[Yun Zhang](https://codec.siat.ac.cn/yunzhang/), Shisheng Zhang(https://orcid.org/0009-0006-5527-7308), [Na Li](https://hpcc.siat.ac.cn/homepage/lina.html), Chunlin Fan, [Raouf Hamzaoui](http://www.tech.dmu.ac.uk/~hamzaoui/) \
*IEEE Transactions on Circuits and Systems for Video Technology (Early Access)*

## Abstract
The Picture-Wise Just Noticeable Difference (PW-JND) represents the visibility threshold of human vision when viewing distorted images. The PW-JND plays an important role in perceptual image processing and compression. However, predicting the PW-JND is challenging due to its dependence on image content, viewing conditions, and the viewer. In this paper, we propose a visual perception-assisted deep PW-JND (VP-JND) prediction model for image compression that combines data-driven methods with the perceptual mechanisms of human vision. First, we identify a correlation between PW-JND and conventional pixel-wise JND. Based on this observation, we design the VP-JND model, consisting of a pixel-wise JND model, a deep binary classifier (VP-JNDnet) and a binary block search algorithm for refining predictions. VP-JNDnet exploits the pixel-wise JND map of the original image to predict whether a compressed image is perceptually lossless. In addition, the model incorporates visual importance of content and regions by using a mixed attention module and calculating perceptual loss during training. Experimental results show that VP-JND achieved an average precision of 94.82% and a mean absolute difference of 3.92 in predicting the JPEG quality factor corresponding to the PW-JND on the MCL-JCI dataset, outperforming state-of-the-art JND models. When applied to perceptual lossless image coding, the predicted PW-JND enabled average bit rate savings of 89.35% for JPEG compression on MCL-JCI and 85.46%/41.13% for JPEG/BPG compression on KonJND-1k. These savings were relative to images compressed at the lowest distortion level. The source codes and trained models are publicly available at https://github.com/SYSU-Video/VP-JND.
