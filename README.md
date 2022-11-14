# CabViT: Cross Attention among Blocks for Vision Transformer
Official PyTorch implementation of **CabViT**

---
<p align="center">
<img src="https://s1.ax1x.com/2022/11/14/zA1VYV.png" width=100% height=100% 
class="center">
</p>

Models
<p align="center">
<img src="https://s1.ax1x.com/2022/11/14/zA1CLj.md.png" width=60% height=60% 
class="center">
</p>

<p align="center">
<img src="https://s1.ax1x.com/2022/11/14/zA19yQ.png" width=60% height=60% 
class="center">
</p>

## Introduction
Since the vision transformer (ViT) has achieved impressive performance in image classification, an increasing number of researchers pay their attentions to designing more efficient vision transformer models. A general research line is reducing computational cost of self attention modules by adopting sparse attention or using local attention windows. In contrast, we propose to design high performance transformer based architectures by densifying the attention pattern. Specifically, we propose cross attention among blocks of ViT (CabViT), which uses tokens from previous blocks in the same stage as extra input to the multi-head attention of transformers. The proposed CabViT enhances the interactions of tokens across blocks with potentially different semantics, and encourages more information flows to the lower levels, which together improves model performance and model convergence with limited extra cost. Based on the proposed CabViT, we design a series of CabViT models which achieve the best trade-off between model size, computational cost and accuracy. For instance without the need of knowledge distillation to strength the training, CabViT achieves 83.0% top-1 accuracy on Imagenet with only 16.3 million parameters and about 3.9G FLOPs, saving almost half parameters and 13% computational cost while gaining 0.9% higher accuracy compared with ConvNext, use 52% of parameters but gaining 0.6% accuracy compared with distilled EfficientFormer
