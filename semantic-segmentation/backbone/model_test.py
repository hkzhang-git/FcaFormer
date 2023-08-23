import torch

from semantic_segmentation.backbone.cabvit_512 import CabViT_SH_512
# from semantic_segmentation.backbone.cabvit_224 import CabViT_SH_224

input=torch.randn(3, 3, 512, 512)
model=CabViT_SH_512()
output = model(input)
print('done')
