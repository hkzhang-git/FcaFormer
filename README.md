# Fcaformer: Forward Cross Attention in Hybrid Vision Transformer
Official PyTorch implementation of **FcaFormer**

---
Comparison with SOTA models
<p align="center">
<img src="https://s1.ax1x.com/2023/03/20/pptbZ7Q.png" width=100% height=100% 
class="center">
</p>

Architecture
<p align="center">
<img src="https://s1.ax1x.com/2023/03/20/pptbmkj.png" width=100% height=100% 
class="center">
</p>
A FcaFormer stage which consists of L FcaFormer blocks. Compared with standard ViT blocks, we add token merge and enhancement (TME) part, which uses long stride large kernel convolutions to merge tokens spatially as cross tokens, and small kernel convolutions to further enhance tokens for channel mixing (FFN). The cross tokens are then used in later blocks as extra tokens for multi-head attention, after being calibrated by multiplying them with learned scaling factors.

<p align="center">
<img src="https://s1.ax1x.com/2023/03/20/pptbnts.png" width=60% height=60% 
class="center">
</p>
The overall structure of FcaFormer Mobels for image classification tasks. (a) Plain FcaFormer model. This model is directly modified from the DeiT structure. (b) Hybrid FcaFormer model. The ConvNet stages are composed of ConvNext blocks.

## Introduction
Currently, one main research line in designing a more efficient vision transformer is reducing the computational cost of self attention modules by adopting sparse attention or using local attention windows. In contrast, we propose a different approach that aims to improve the performance of transformer-based architectures by densifying the attention pattern. Specifically, we proposed forward cross attention for hybrid vision transformer (FcaFormer), where tokens from previous blocks in the same stage are secondary used. To achieve this, the FcaFormer leverages two innovative components: learnable scale factors (LSFs) and a token merge and enhancement module (TME). The LSFs enable efficient processing of cross tokens, while the TME generates representative cross tokens. By integrating these components, the proposed FcaFormer enhances the interactions of tokens across blocks with potentially different semantics, and encourages more information flows to the lower levels. Based on the forward cross attention (Fca), we have designed a series of FcaFormer models that achieve the best trade-off between model size, computational cost, memory cost, and accuracy. For example, without the need for knowledge distillation to strengthen training, our FcaFormer achieves 83.1% top-1 accuracy on Imagenet with only 16.3 million parameters and about 3.6 billion MACs. This saves almost half of the parameters and a few computational costs while achieving 0.7% higher accuracy compared to distilled EfficientFormer. 

## Image classification on ImageNet 1k

| Models | #params (M) | MACs(M)| Top1 acc |
|:---:|:---:|:---:|:---:|
| Deit-T       | 5.5   | -   |  72.2 |
| FcaFormer-L0 | 5.9   | -   |  74.3 |
|Swin-1G       | 6.3   | 1.5 |  78.4 |
|FcaFormer-L1  | 6.2   | 1.4 |  80.3 |
|ConvNext-Tiny | 29    | 4.5 |  82.1 |
|Swin-Tiny     | 29    | 4.5 |  81.3 |
|FcaFormer-L2  | 16.3  | 3.6 |  83.1 |
 
 ## Semantic segmentation on ADE20K

| Method | Backbone | mIOU|  #params  |  MACs |
|:---:|:---:|:---:|:---:|:---:|
| DNL      | ResNet-101 |  46.0    | 69    | 1249  |
| OCRNet   | ResNet-101 |  45.3    | 56    | 923   |
|UperNet   | ResNet-101 |  44.9    | 86    | 1029  |
|UperNet   | DeiT III (ViT-S) | 46.8 | 42  | 588   |
|UperNet   | Swin-T     | 45.8     | 60    |  945  |
|UperNet   | ConNext-T  | 46.7     | 60    | 939   |
|UperNet   | FcaFormer-L2 | 47.6   | 46    | 730   |
 
## Object detection on COCO


## Test on edge device
