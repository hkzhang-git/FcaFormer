# from .darknet import Darknet
# from .detectors_resnet import DetectoRS_ResNet
# from .detectors_resnext import DetectoRS_ResNeXt
# from .hourglass import HourglassNet
# from .hrnet import HRNet
# from .regnet import RegNet
# from .res2net import Res2Net
# from .resnest import ResNeSt
# from .resnext import ResNeXt
# from .ssd_vgg import SSDVGG
# from .trident_resnet import TridentResNet
from .resnet import ResNet, ResNetV1d
from .swin_transformer import SwinTransformer
from .convnext import ConvNeXt
from .fcaformer import FcaFormer_SH_512

# __all__ = [
#     'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
#     'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet',
#     'ResNeSt', 'TridentResNet', 'SwinTransformer', 'ConvNeXt'
# ]

__all__ = [
    'ResNet', 'ResNetV1d', 'SwinTransformer', 'ConvNeXt', 'FcaFormer_SH_512'
]


